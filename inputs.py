from typing import Any, Generic, Sequence, Union, Type, TypeVar
from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
import torch
from torch import nn

Path = Sequence[Union[str, int]]
PathOrRemainder = Union[Path, int]

T = TypeVar('T')

class Space(Generic[T], ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> T:
        """Sample a random value from this space."""
        pass


class ScalarSpace(Generic[T], Space[T]):
    """Embeds a scalar tensor into a tensor."""

    @abstractmethod
    def build_embed(self, dim: int) -> nn.Module:
        pass


class MaskedSpace(Generic[T], Space[T]):
    """
    Embeds an object into an (embedding, mask) pair of tensors and supports finding
    the path to a specific index.

    embedding has shape [batch, seq, dim]
    mask has shape [batch, seq]
    """

    @abstractmethod
    def build_embed(self, dim: int) -> nn.Module:
        pass

    @abstractmethod
    def try_ix_to_path(self, x: T, ix: int) -> PathOrRemainder:
        """Try to convert a flat index to a hierarchical path."""
        pass

    def ix_to_path(self, x: T, ix: int) -> Path:
        """Convert a flat index to a hierarchical path."""
        p = self.try_ix_to_path(x, ix)
        if isinstance(p, int):
            raise IndexError(f"Index {ix} is out of bounds")
        else:
            return p

    def length(self, x: T) -> int:
        """Get the total number of indexable elements in x."""
        INF = 1_000_000_000
        rem = self.try_ix_to_path(x, INF)
        assert isinstance(rem, int)
        return INF - rem

    @abstractmethod
    def path_to_ix(self, x: T, path: Path) -> int:
        """Convert a hierarchical path to a flat index."""
        pass


TEnum = TypeVar('TEnum', bound=IntEnum)

class EnumSpace(ScalarSpace[TEnum]):
    def __init__(self, enum_class: Type[TEnum]):
        self.enum_class = enum_class
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return nn.Embedding(len(self.enum_class), dim)

    def sample(self, rng: np.random.Generator) -> TEnum:
        return rng.choice(list(self.enum_class))


class IntSpace(ScalarSpace[int]):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return nn.Embedding(self.limit, dim)

    def sample(self, rng: np.random.Generator) -> int:
        return rng.integers(0, self.limit)


class SequenceEmbedding(nn.Module):
    """Module for embedding sequences using an element embedding."""
    def __init__(self, element_embedding: nn.Module):
        super().__init__()
        self.element_embedding = element_embedding
    
    def forward(self, xs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.element_embedding(xs["value"]), xs["mask"]

class SequenceSpace(MaskedSpace[Sequence[T]]):
    def __init__(self, element_space: ScalarSpace[T]):
        self.element_space = element_space
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return SequenceEmbedding(self.element_space.build_embed(dim))

    def sample(self, rng: np.random.Generator) -> Sequence[T]:
        length = int(rng.exponential(10))
        return [self.element_space.sample(rng) for _ in range(length)]


    def try_ix_to_path(self, xs: dict, ix: int) -> PathOrRemainder:
        l = xs["value"].size(1)
        if ix < l:
            return [ix]
        else:
            return ix - l

    def path_to_ix(self, xs: dict, path: Path) -> int:
        (i,) = path
        if i >= xs["value"].size(1):
            raise IndexError(f"Path index {i} out of bounds")
        return i

class SinusoidalEmbedding(nn.Module):
    """Standalone sinusoidal embedding for numerical features."""
    def __init__(self, dim: int, n_features: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('inv_freq', torch.exp(torch.arange(half_dim) * -emb) * 10)
        self.out_dim = dim * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for numerical values.
        x: tensor of integers to embed [batch_size, n_features]
        Returns: [batch_size, n_features * dim] tensor
        """
        # [batch_size, n_features, half_dim]
        emb = x.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)

        # [batch_size, n_features, dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # [batch_size, n_features * dim]
        return emb.reshape(x.shape[0], -1)

class FixedVecSpace(MaskedSpace[np.ndarray]):
    def __init__(self, limits: list[int]):
        self.limits = limits
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        limits = self.limits  # Capture in closure
        
        class FixedVecEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_embed = SinusoidalEmbedding(dim, len(limits))
                self.proj = nn.Linear(self.num_embed.out_dim, dim)
            
            def forward(self, xs):
                # xs is [batch_size, n_features]
                embedded = self.proj(self.num_embed(xs))  # [batch_size, dim]
                embedded = embedded.unsqueeze(1)  # [batch_size, 1, dim]
                mask = torch.zeros(embedded.shape[0], 1, dtype=torch.bool, device=embedded.device)
                return embedded, mask
        
        return FixedVecEmbedding()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.randint(0, limit) for limit in self.limits])


    def try_ix_to_path(self, x: np.ndarray, ix: int) -> PathOrRemainder:
        if ix == 0:
            return []
        else:
            return ix - 1

    def path_to_ix(self, x: np.ndarray, path: Path) -> int:
        if len(path) == 0:
            return 0
        else:
            raise IndexError(f"Path {path} is out of bounds")

class TupleAddEmbedding(nn.Module):
    """Module for embedding tuples by adding component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleList):
        super().__init__()
        self.component_embeddings = component_embeddings
    
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # xs shape: [batch, seq, n_components] or [batch, n_components]
        # Split along last dimension and apply embeddings
        component_outputs = []
        for i, emb in enumerate(self.component_embeddings):
            component_input = xs[..., i]  # [batch, seq] or [batch]
            component_outputs.append(emb(component_input))
        
        # Sum the embeddings
        return torch.sum(torch.stack(component_outputs, dim=0), dim=0)
class TupleAddSpace(ScalarSpace[tuple]):
    def __init__(self, *spaces: ScalarSpace[Any]):
        self.spaces = spaces
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return TupleAddEmbedding(nn.ModuleList([space.build_embed(dim) for space in self.spaces]))

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)

class TupleConcatEmbedding(nn.Module):
    """Module for embedding tuples by concatenating component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleList):
        super().__init__()
        self.component_embeddings = component_embeddings
    
    def forward(self, xs: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings, masks = zip(*[emb(x) for emb, x in zip(self.component_embeddings, xs)])
        return torch.cat(embeddings, dim=1), torch.cat(masks, dim=1)

class DictEmbedding(nn.Module):
    """Module for embedding dictionaries by concatenating component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleDict, spaces: dict):
        super().__init__()
        self.component_embeddings = component_embeddings
        self.spaces = spaces
    
    def forward(self, xs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings, masks = zip(*[self.component_embeddings[k](xs[k]) for k in self.spaces.keys()])
        return torch.cat(embeddings, dim=1), torch.cat(masks, dim=1)

class TupleConcatSpace(MaskedSpace[tuple]):
    def __init__(self, *spaces: MaskedSpace[Any]):
        self.spaces = spaces
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return TupleConcatEmbedding(nn.ModuleList([space.build_embed(dim) for space in self.spaces]))

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)


    def try_ix_to_path(self, x: tuple, ix: int) -> PathOrRemainder:
        for i, (space, elem) in enumerate(zip(self.spaces, x)):
            p = space.try_ix_to_path(elem, ix)
            if isinstance(p, int):
                ix = p
            else:
                return [i] + p
        return ix

    def path_to_ix(self, xs: tuple, path: Path) -> int:
        i, *subpath = path
        if i >= len(xs):
            raise IndexError(f"Path index {i} out of bounds")
        ix = 0
        for (space, x) in zip(self.spaces[:i], xs[:i]):
            ix += space.length(x)
        return ix + self.spaces[i].path_to_ix(xs[i], subpath)

class DictSpace(MaskedSpace[dict[str, Any]]):
    def __init__(self, spaces: dict[str, MaskedSpace[Any]]):
        self.spaces = spaces
        super().__init__()
    
    def build_embed(self, dim: int) -> nn.Module:
        return DictEmbedding(nn.ModuleDict({k: space.build_embed(dim) for k, space in self.spaces.items()}), self.spaces)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {k: v.sample(rng) for k, v in self.spaces.items()}


    def try_ix_to_path(self, x: dict[str, Any], ix: int) -> PathOrRemainder:
        for k, v in x.items():
            p = self.spaces[k].try_ix_to_path(v, ix)
            if isinstance(p, int):
                ix = p
            else:
                return [k] + p
        return ix

    def path_to_ix(self, x: dict[str, Any], path: Path) -> int:
        k, *subpath = path
        return self.spaces[k].path_to_ix(x[k], subpath)
