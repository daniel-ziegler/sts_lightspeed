from __future__ import annotations

from typing import Any, Generic, Sequence, Union, Type, TypeVar, Callable
from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
import torch
from torch import nn

Path = Sequence[Union[str, int]]
PathOrRemainder = Union[Path, int]

class EmbedCache(dict[tuple, nn.Module]):
    def __init__(self):
        super().__init__()
    
    def build(self, space: Space, dim: int, make: Callable[[], nn.Module]) -> nn.Module:
        if (space, dim) not in self:
            embed = make()
            self[(space, dim)] = embed
        return self[(space, dim)]

T = TypeVar('T')

class Space(Generic[T], ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> T:
        """Sample a random value from this space."""
        pass

    @abstractmethod
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        pass

class ScalarSpace(Generic[T], Space[T]):
    """Embeds a scalar tensor into a tensor."""


class MaskedSpace(Generic[T], Space[T]):
    """
    Embeds an object into an (embedding, mask) pair of tensors and supports finding
    the path to a specific index.

    embedding has shape [batch, seq, dim]
    mask has shape [batch, seq]
    """

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

class EnumEmbedding(nn.Module):
    def __init__(self, enum_class: Type[TEnum], dim: int):
        super().__init__()
        self.enum_class = enum_class
        self.embedding = nn.Embedding(len(enum_class), dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert ((0 <= x) & (x < len(self.enum_class))).all(), f"Enum values {x} out of bounds for {self.enum_class}"
        return self.embedding(x)

class EnumSpace(ScalarSpace[TEnum]):
    def __init__(self, enum_class: Type[TEnum]):
        self.enum_class = enum_class
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        return cache.build(self, dim, lambda: EnumEmbedding(self.enum_class, dim))


    def sample(self, rng: np.random.Generator) -> TEnum:
        values = list(self.enum_class)
        return rng.choice(values)


class IntSpace(ScalarSpace[int]):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        return cache.build(self, dim, lambda: nn.Embedding(self.limit, dim))

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
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        return SequenceEmbedding(self.element_space.build_embed(dim, cache))

    def sample(self, rng: np.random.Generator) -> Sequence[T]:
        length = int(rng.exponential(10))
        return [self.element_space.sample(rng) for _ in range(length)]


    def try_ix_to_path(self, xs: dict, ix: int) -> PathOrRemainder:
        t = xs["value"]
        while isinstance(t, dict):
            t = next(iter(t.values()))
        l = t.size(1)
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


class FixedVecEmbedding(nn.Module):
    def __init__(self, dim: int, limits: list[int]):
        super().__init__()
        self.limits = limits
        self.num_embed = SinusoidalEmbedding(dim, len(limits))
        self.proj = nn.Linear(self.num_embed.out_dim, dim)
    
    def forward(self, xs):
        # xs is [..., n_features]
        assert xs.shape[-1] == len(self.limits), f"FixedVecSpace limits {self.limits} do not match input shape {xs.shape}"
        
        original_shape = xs.shape
        
        xs_reshaped = xs.reshape(-1, xs.shape[-1])
        
        embedded = self.proj(self.num_embed(xs_reshaped))  # [*, dim]
        
        output_shape = list(original_shape[:-1]) + [embedded.shape[-1]]
        return embedded.reshape(output_shape)

class FixedVecSpace(ScalarSpace[np.ndarray]):
    def __init__(self, limits: list[int]):
        self.limits = limits
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        return cache.build(self, dim, lambda: FixedVecEmbedding(dim, self.limits))

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.integers(0, limit) for limit in self.limits])


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
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        return TupleAddEmbedding(nn.ModuleList([space.build_embed(dim, cache) for space in self.spaces]))

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)


class ScalarToSequenceEmbedding(nn.Module):
    """Adapter to convert scalar embedding to sequence embedding format."""
    def __init__(self, scalar_embedding: nn.Module):
        super().__init__()
        self.scalar_embedding = scalar_embedding
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is a single value per batch item: [batch_size]
        # Convert to scalar embedding: [batch_size, dim]
        embedding = self.scalar_embedding(x)
        
        # Add sequence dimension: [batch_size, 1, dim]
        embedding = embedding.unsqueeze(1)
        
        # Create mask: [batch_size, 1] - all False (not masked)
        mask = torch.zeros(embedding.shape[0], 1, dtype=torch.bool, device=embedding.device)
        
        return embedding, mask


class ScalarToSequenceSpace(MaskedSpace[Any]):
    """Adapter to convert ScalarSpace to MaskedSpace for use in DictSpace."""
    def __init__(self, scalar_space: ScalarSpace[Any]):
        self.scalar_space = scalar_space
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        scalar_embedding = self.scalar_space.build_embed(dim, cache)
        return ScalarToSequenceEmbedding(scalar_embedding)
    
    def sample(self, rng: np.random.Generator) -> Any:
        return self.scalar_space.sample(rng)
    
    def try_ix_to_path(self, x: Any, ix: int) -> PathOrRemainder:
        # For single values, only index 0 is valid
        if ix == 0:
            return []
        return ix - 1
    
    def path_to_ix(self, x: Any, path: Path) -> int:
        # For single values, only empty path is valid (index 0)
        if len(path) == 0:
            return 0
        raise ValueError(f"Invalid path {path} for scalar value")


class DictAddEmbedding(nn.Module):
    """Module for embedding dictionaries by adding component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleDict):
        super().__init__()
        self.component_embeddings = component_embeddings
    
    def forward(self, xs: dict) -> torch.Tensor:
        # Apply embeddings to each component and sum them
        component_outputs = []
        for key, emb in self.component_embeddings.items():
            component_input = xs[key]
            component_outputs.append(emb(component_input))
        
        # Sum the embeddings
        return torch.sum(torch.stack(component_outputs, dim=0), dim=0)


class DictAddSpace(ScalarSpace[dict]):
    """Space for dictionaries where component embeddings are added together."""
    def __init__(self, spaces: dict[str, ScalarSpace[Any]]):
        self.spaces = spaces
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        def make():
            component_embeddings = nn.ModuleDict({k: space.build_embed(dim, cache) for k, space in self.spaces.items()})
            return DictAddEmbedding(component_embeddings)
        return cache.build(self, dim, make)
    
    def sample(self, rng: np.random.Generator) -> dict:
        return {k: space.sample(rng) for k, space in self.spaces.items()}


class TupleConcatEmbedding(nn.Module):
    """Module for embedding tuples by concatenating component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleList, dim: int):
        super().__init__()
        self.component_embeddings = component_embeddings
        self.position_embedding = nn.Embedding(len(component_embeddings), dim)
    
    def forward(self, xs: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings, masks = zip(*[emb(x) for emb, x in zip(self.component_embeddings, xs)])
        
        # Create position indices for all components at once
        position_indices = []
        for i, mask in enumerate(masks):
            batch_size, seq_len = mask.shape
            position_indices.append(torch.full((batch_size, seq_len), i, dtype=torch.long, device=mask.device))
        
        # Concatenate position indices and get embeddings in one call
        all_position_indices = torch.cat(position_indices, dim=1)
        position_embeddings = self.position_embedding(all_position_indices)
        
        # Add position embeddings to content embeddings
        all_embeddings = torch.cat(embeddings, dim=1)
        return all_embeddings + position_embeddings, torch.cat(masks, dim=1)


class DictEmbedding(nn.Module):
    """Module for embedding dictionaries by concatenating component embeddings."""
    def __init__(self, component_embeddings: nn.ModuleDict, spaces: dict, dim: int):
        super().__init__()
        self.component_embeddings = component_embeddings
        self.spaces = spaces
        self.position_embedding = nn.Embedding(len(spaces), dim)
        self.key_to_index = {k: i for i, k in enumerate(spaces.keys())}
    
    def forward(self, xs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings, masks = zip(*[self.component_embeddings[k](xs[k]) for k in self.spaces.keys()])
        
        # Create position indices for all components at once
        position_indices = []
        for k, mask in zip(self.spaces.keys(), masks):
            batch_size, seq_len = mask.shape
            key_idx = self.key_to_index[k]
            position_indices.append(torch.full((batch_size, seq_len), key_idx, dtype=torch.long, device=mask.device))
        
        # Concatenate position indices and get embeddings in one call
        all_position_indices = torch.cat(position_indices, dim=1)
        position_embeddings = self.position_embedding(all_position_indices)
        
        # Add position embeddings to content embeddings
        all_embeddings = torch.cat(embeddings, dim=1)
        return all_embeddings + position_embeddings, torch.cat(masks, dim=1)


class TupleConcatSpace(MaskedSpace[tuple]):
    def __init__(self, *spaces: MaskedSpace[Any]):
        self.spaces = spaces
        super().__init__()
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        def make():
            return TupleConcatEmbedding(nn.ModuleList([space.build_embed(dim, cache) for space in self.spaces]), dim)
        return cache.build(self, dim, make)

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
    
    def build_embed(self, dim: int, cache: EmbedCache) -> nn.Module:
        def make():
            return DictEmbedding(nn.ModuleDict({k: space.build_embed(dim, cache) for k, space in self.spaces.items()}), self.spaces, dim)
        return cache.build(self, dim, make)

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
        ix = 0
        # Accumulate lengths from all keys before the target key
        for key in self.spaces.keys():
            if key == k:
                break
            ix += self.spaces[key].length(x[key])
        return ix + self.spaces[k].path_to_ix(x[k], subpath)
