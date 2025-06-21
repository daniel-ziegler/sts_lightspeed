from typing import Any, Generic, Sequence, Union, Type, TypeVar
from abc import ABC, abstractmethod

import numpy as np
from enum import IntEnum
import torch
from torch import nn

Path = Sequence[Union[str, int]]
PathOrRemainder = Union[Path, int]

T = TypeVar('T')

class Space(Generic[T], nn.Module, ABC):
    """Unified Space class that combines data structure definition with neural network embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> T:
        """Sample a random value from this space."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tensor into representation space."""
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

class EnumSpace(Space[TEnum]):
    def __init__(self, enum_class: Type[TEnum], dim: int):
        self.enum_class = enum_class
        super().__init__(dim)
        self.embedding = nn.Embedding(len(self.enum_class), self.dim)

    def sample(self, rng: np.random.Generator) -> TEnum:
        return rng.choice(list(self.enum_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def try_ix_to_path(self, x: TEnum, ix: int) -> PathOrRemainder:
        if ix == 0:
            return []
        else:
            return ix - 1

    def path_to_ix(self, x: TEnum, path: Path) -> int:
        if len(path) == 0:
            return 0
        else:
            raise IndexError(f"Path {path} is out of bounds")

class IntSpace(Space[int]):
    def __init__(self, limit: int, dim: int):
        self.limit = limit
        super().__init__(dim)
        self.embedding = nn.Embedding(self.limit, self.dim)

    def sample(self, rng: np.random.Generator) -> int:
        return rng.integers(0, self.limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def try_ix_to_path(self, x: int, ix: int) -> PathOrRemainder:
        if ix == 0:
            return []
        else:
            return ix - 1

    def path_to_ix(self, x: int, path: Path) -> int:
        if len(path) == 0:
            return 0
        else:
            raise IndexError(f"Path {path} is out of bounds")


class SequenceSpace(Space[Sequence[T]]):
    def __init__(self, element_space: Space[T], dim: int):
        self.element_space = element_space
        super().__init__(dim)
        self.element_embedding = self.element_space

    def sample(self, rng: np.random.Generator) -> Sequence[T]:
        length = int(rng.exponential(10))
        return [self.element_space.sample(rng) for _ in range(length)]

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.element_embedding(xs)

    def try_ix_to_path(self, xs: Sequence[T], ix: int) -> PathOrRemainder:
        for i, x in enumerate(xs):
            p = self.element_space.try_ix_to_path(x, ix)
            if isinstance(p, int):
                ix -= p
            else:
                return [i] + p
        return ix

    def path_to_ix(self, xs: Sequence[T], path: Path) -> int:
        i, *subpath = path
        ix = 0
        for x in xs[:i]:
            ix += self.element_space.length(x)
        return ix + self.element_space.path_to_ix(xs[i], subpath)

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

class FixedVecSpace(Space[np.ndarray]):
    def __init__(self, limits: list[int], dim: int):
        self.limits = limits
        super().__init__(dim)
        self.num_embed = SinusoidalEmbedding(self.dim, len(self.limits))
        self.proj = nn.Linear(self.num_embed.out_dim, self.dim)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.randint(0, limit) for limit in self.limits])

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.num_embed(xs))

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

class TupleAddSpace(Space[tuple]):
    def __init__(self, *spaces: Space[Any], dim: int):
        self.spaces = spaces
        super().__init__(dim)
        self.component_spaces = nn.ModuleList(self.spaces)

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)

    def forward(self, xs: tuple) -> torch.Tensor:
        return torch.sum(torch.stack([space(x) for space, x in zip(self.spaces, xs)], dim=0), dim=0)

    def try_ix_to_path(self, x: tuple, ix: int) -> PathOrRemainder:
        for i, (space, elem) in enumerate(zip(self.spaces, x)):
            p = space.try_ix_to_path(elem, ix)
            if isinstance(p, int):
                ix -= p
            else:
                return [i] + p
        return ix

    def path_to_ix(self, x: tuple, path: Path) -> int:
        i, *subpath = path
        ix = 0
        for j, (space, elem) in enumerate(zip(self.spaces, x)):
            if j < i:
                ix += space.length(elem)
            elif j == i:
                return ix + space.path_to_ix(elem, subpath)
        raise IndexError(f"Path index {i} out of bounds")

class TupleConcatSpace(Space[tuple]):
    def __init__(self, *spaces: Space[Any], dim: int):
        self.spaces = spaces
        super().__init__(dim)
        self.component_spaces = nn.ModuleList(self.spaces)

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)

    def forward(self, xs: tuple) -> torch.Tensor:
        return torch.cat([space(x) for space, x in zip(self.spaces, xs)], dim=-2)

    def try_ix_to_path(self, x: tuple, ix: int) -> PathOrRemainder:
        for i, (space, elem) in enumerate(zip(self.spaces, x)):
            p = space.try_ix_to_path(elem, ix)
            if isinstance(p, int):
                ix -= p
            else:
                return [i] + p
        return ix

    def path_to_ix(self, x: tuple, path: Path) -> int:
        i, *subpath = path
        ix = 0
        for j, (space, elem) in enumerate(zip(self.spaces, x)):
            if j < i:
                ix += space.length(elem)
            elif j == i:
                return ix + space.path_to_ix(elem, subpath)
        raise IndexError(f"Path index {i} out of bounds")

class DictSpace(Space[dict[str, Any]]):
    def __init__(self, spaces: dict[str, Space[Any]], dim: int):
        self.spaces = spaces
        super().__init__(dim)
        self.component_spaces = nn.ModuleDict(self.spaces)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {k: v.sample(rng) for k, v in self.spaces.items()}

    def forward(self, xs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.spaces[k](x) for k, x in xs.items()], dim=-2)

    def try_ix_to_path(self, x: dict[str, Any], ix: int) -> PathOrRemainder:
        for k, v in x.items():
            p = self.spaces[k].try_ix_to_path(v, ix)
            if isinstance(p, int):
                ix -= p
            else:
                return [k] + p
        return ix

    def path_to_ix(self, x: dict[str, Any], path: Path) -> int:
        k, *subpath = path
        return self.spaces[k].path_to_ix(x[k], subpath)
