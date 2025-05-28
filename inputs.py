from typing import Any, Generic, Sequence, Union, Type, TypeVar, TypeVarTuple, Unpack


import numpy as np
from enum import IntEnum
import torch
from torch import nn

import abc

Path = Sequence[Union[str, int]]
PathOrRemainder = Union[Path, int]

T = TypeVar('T')

class Space(Generic[T], abc.ABC):
    @abc.abstractmethod
    def sample(self, rng: np.random.Generator) -> T:
        pass

    @abc.abstractmethod
    def build_embedding(self, dim: int) -> nn.Module:
        """Builds a module that embeds into a tensor of shape (batch, seq, dim)"""
        pass

    @abc.abstractmethod
    def try_ix_to_path(self, x: T, ix: int) -> PathOrRemainder:
        pass

    def ix_to_path(self, x: T, ix: int) -> Path:
        p = self.try_ix_to_path(x, ix)
        if isinstance(p, int):
            raise IndexError(f"Index {ix} is out of bounds")
        else:
            return p

    def length(self, x: T) -> int:
        INF = 1_000_000_000
        rem = self.try_ix_to_path(x, INF)
        assert isinstance(rem, int)
        return INF - rem

    @abc.abstractmethod
    def path_to_ix(self, xspath: Path) -> int:
        pass


TEnum = TypeVar('TEnum', bound=IntEnum)

class EnumSpace(Space[TEnum]):
    def __init__(self, enum_class: Type[TEnum]):
        self.enum_class = enum_class

    def sample(self, rng: np.random.Generator) -> TEnum:
        return rng.choice(list(self.enum_class))

    def build_embedding(self, dim: int) -> nn.Module:
        return nn.Embedding(len(self.enum_class), dim)

class IntSpace(Space[int]):
    def __init__(self, limit: int):
        self.limit = limit

    def sample(self, rng: np.random.Generator) -> int:
        return rng.integers(0, self.limit)

    def build_embedding(self, dim: int) -> nn.Module:
        return nn.Embedding(self.limit, dim)


class SequenceEmbedding(nn.Module):
    def __init__(self, space: Space[T], dim: int):
        super().__init__()
        self.embedding = space.build_embedding(dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        # TODO dims
        return self.embedding(xs)

class SequenceSpace(Space[Sequence[T]]):
    def __init__(self, space: Space[T]):
        self.space = space

    def sample(self, rng: np.random.Generator) -> Sequence[T]:
        length = int(rng.exponential(10))
        return [self.space.sample(rng) for _ in range(length)]

    def build_embedding(self, dim: int) -> nn.Module:
        return SequenceEmbedding(self.space, dim)

    def try_ix_to_path(self, xs: Sequence[T], ix: int) -> PathOrRemainder:
        for i, x in enumerate(xs):
            p = self.space.try_ix_to_path(x)
            if isinstance(p, int):
                ix -= p
            else:
                return [i] + p
        return ix

    def path_to_ix(self, xs: Sequence[T], path: Path) -> int:
        i, *subpath = path
        for x in xs[:i]:
            ix += self.space.length(x)
        return ix + self.space.path_to_ix(xs[i], subpath)

class SinusoidalEmbedding(nn.Module):
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
    def __init__(self, limits: list[int], dim: int):
        super().__init__()
        self.limits = limits
        # TODO actually use limits to rescale inputs?
        self.num_embed = SinusoidalEmbedding(dim, len(limits))
        self.proj = nn.Linear(self.num_embed.out_dim, dim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.num_embed(xs))


class FixedVecSpace(Space[np.ndarray]):
    def __init__(self, limits: list[int]):
        self.limits = limits

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.randint(0, limit) for limit in self.limits])

    def build_embedding(self, dim: int) -> nn.Module:
        return FixedVecEmbedding(self.limits, dim)

    def try_ix_to_path(self, x: np.ndarray, ix: int) -> PathOrRemainder:
        if ix == 0:
            return []
        else:
            raise IndexError(f"Index {ix} is out of bounds")

    def path_to_ix(self, x: np.ndarray, path: Path) -> int:
        if len(path) == 0:
            return 0
        else:
            raise IndexError(f"Path {path} is out of bounds")

class TupleAddEmbedding(nn.Module):
    def __init__(self, spaces: list[Space[Any]], dim: int):
        super().__init__()
        self.embeddings = [space.build_embedding(dim) for space in spaces]

    def forward(self, xs: tuple) -> torch.Tensor:
        return torch.sum(torch.stack([embedding(x) for embedding, x in zip(self.embeddings, xs)], dim=0), dim=0)

class TupleAddSpace(Space[tuple]):
    def __init__(self, *spaces: Space[Any]):
        self.spaces = spaces

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)

    def build_embedding(self, dim: int) -> nn.Module:
        return TupleAddEmbedding(self.spaces, dim)

class TupleConcatEmbedding(nn.Module):
    def __init__(self, spaces: list[Space[Any]], dim: int):
        super().__init__()
        self.embeddings = [space.build_embedding(dim) for space in spaces]

    def forward(self, xs: tuple) -> torch.Tensor:
        return torch.cat([embedding(x) for embedding, x in zip(self.embeddings, xs)], dim=-2)

class TupleConcatSpace(Space[tuple]):
    def __init__(self, *spaces: Space[Any]):
        self.spaces = spaces

    def sample(self, rng: np.random.Generator) -> tuple:
        return tuple(space.sample(rng) for space in self.spaces)

    def build_embedding(self, dim: int) -> nn.Module:
        return TupleConcatEmbedding(self.spaces, dim)

class DictEmbedding(nn.Module):
    def __init__(self, spaces: dict[str, Space[Any]], dim: int):
        super().__init__()
        self.embeddings = {k: v.build_embedding(dim) for k, v in spaces.items()}

    def forward(self, xs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.embeddings[k](x) for k, x in xs.items()], dim=-2)
    
class DictSpace(Space[dict[str, Any]]):
    def __init__(self, spaces: dict[str, Space[Any]]):
        self.spaces = spaces

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {k: v.sample(rng) for k, v in self.spaces.items()}

    def build_embedding(self, dim: int) -> nn.Module:
        return DictEmbedding(self.spaces, dim)

    def try_ix_to_path(self, x: dict[str, Any], ix: int) -> PathOrRemainder:
        for k, v in x.items():
            p = self.spaces[k].try_ix_to_path(v)
            if isinstance(p, int):
                ix -= p
            else:
                return [k] + p
        return ix

    def path_to_ix(self, x: dict[str, Any], path: Path) -> int:
        k, *subpath = path
        return self.spaces[k].path_to_ix(x[k], subpath)
