class Potion:

    # CommunicationMod's `requires_target` is wrong for some potions: AOE/no-target potions such as
    # Explosive Potion and Smoke Bomb are reported as requires_target=true. Only the single-target
    # attack/debuff potions actually need a target, so requires_target is derived from this
    # authoritative set (the game's genuinely-targeted potions) instead of trusting the raw field.
    TARGETED_POTION_IDS = frozenset({"FearPotion", "Fire Potion", "Poison Potion", "Weak Potion"})

    def __init__(self, potion_id, name, can_use, can_discard, requires_target, price=0):
        self.potion_id = potion_id
        self.name = name
        self.can_use = can_use
        self.can_discard = can_discard
        self.requires_target = requires_target
        self.price = price

    def __eq__(self, other):
        return other.potion_id == self.potion_id

    @classmethod
    def from_json(cls, json_object):
        potion_id = json_object.get("id")
        return cls(
            potion_id=potion_id,
            name=json_object.get("name"),
            can_use=json_object.get("can_use", False),
            can_discard=json_object.get("can_discard", False),
            requires_target=potion_id in cls.TARGETED_POTION_IDS,
            price=json_object.get("price", 0)
        )
