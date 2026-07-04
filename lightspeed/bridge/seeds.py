"""StS seed <-> base-35 string conversion (SeedHelper equivalents)."""





# StS seed alphabet (base-35, no 'O'); matches SeedHelper.CHARACTERS.
_SEED_CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def seed_long_to_string(seed: int) -> str:
    """Convert a numeric game seed (the live game_state's `seed`, a signed int64) to the base-35
    string the game's `start`/`--seed` command expects -- mirrors SeedHelper.getString (unsigned
    base-35). Lets a captured game be replayed deterministically with `comm.py --seed <string>`."""
    n = seed & 0xFFFFFFFFFFFFFFFF  # interpret as unsigned 64-bit, like Long.toUnsignedString
    if n == 0:
        return "0"
    out = []
    while n:
        out.append(_SEED_CHARS[n % 35])
        n //= 35
    return "".join(reversed(out))


def seed_string_to_long(seed_str: str) -> int:
    """Convert a base-35 seed string back to the numeric game seed (signed int64) -- mirrors
    SeedHelper.getLong, inverting `seed_long_to_string`. Accepts lowercase and maps the letter
    'O' to the digit '0' the way the game's seed-entry screen does."""
    n = 0
    for ch in seed_str.strip().upper().replace("O", "0"):
        n = n * 35 + _SEED_CHARS.index(ch)
    n &= 0xFFFFFFFFFFFFFFFF
    return n - 0x10000000000000000 if n >= 0x8000000000000000 else n  # to signed int64
