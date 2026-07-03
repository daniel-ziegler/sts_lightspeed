class Relic:

    def __init__(self, relic_id, name, counter=0, price=0, grayscale=False, activated=None):
        self.relic_id = relic_id
        self.name = name
        self.counter = counter
        self.price = price
        # Forked-CommunicationMod extras (absent on the stock mod / old captures):
        # grayscale -- the relic renders grayed-out; one-shot relics (Centennial Puzzle) set it
        # exactly when their per-combat charge is spent.
        # activated -- the relic's private trigger latch where it has one (Necronomicon: True at
        # turn start = ready, False once the duplication fired this turn). None when not exposed.
        self.grayscale = grayscale
        self.activated = activated

    @classmethod
    def from_json(cls, json_object):
        return cls(json_object["id"], json_object["name"], json_object["counter"], json_object.get("price", 0),
                   json_object.get("grayscale", False), json_object.get("activated"))
