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
