from dataclasses import dataclass


DATA_KEYS = ["case_id"]

TOPIC_NAMES = [
    "applprev",
    "credit_bureau_a",
    "credit_bureau_b",
    "person",
    "debitcard",
    "deposit",
    "other",
    "tax_registry_a",
    "tax_registry_b",
    "tax_registry_c",
    "static",
    "static_cb",
]

@dataclass
class Topic:
    name: str
    depth: int

    def __post_init__(self):
        if self.name not in TOPIC_NAMES:
            raise ValueError(f"Invalid topic: {self.name}. Available topics: {TOPIC_NAMES}")

TOPICS = [
    Topic("applprev", 1),
    Topic("applprev", 2),
    Topic("credit_bureau_a", 1),
    Topic("credit_bureau_a", 2),
    Topic("credit_bureau_b", 1),
    Topic("credit_bureau_b", 2),
    Topic("person", 1),
    Topic("person", 2),
    Topic("debitcard", 1),
    Topic("deposit", 1),
    Topic("other", 1),
    Topic("tax_registry_a", 1),
    Topic("tax_registry_b", 1),
    Topic("tax_registry_c", 1),
    Topic("static", 0),
    Topic("static_cb", 0),
]
