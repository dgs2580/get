"""Unit recognition, canonicalisation, display and conversion."""

UNIT_ALIASES = {
    # mass
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg",
    "kilograms": "kg", "kilogramme": "kg", "kilogrammes": "kg",
    "g": "g", "gm": "g", "gms": "g", "gram": "g", "grams": "g",
    "gramme": "g", "grammes": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    # volume
    "l": "litre", "lt": "litre", "ltr": "litre", "litre": "litre",
    "litres": "litre", "liter": "litre", "liters": "litre",
    "ml": "ml", "millilitre": "ml", "millilitres": "ml",
    "milliliter": "ml", "milliliters": "ml",
    # containers / counts
    "packet": "packet", "packets": "packet", "pack": "packet", "packs": "packet",
    "pkt": "packet", "pkts": "packet", "sachet": "packet", "sachets": "packet",
    "bottle": "bottle", "bottles": "bottle",
    "box": "box", "boxes": "box",
    "dozen": "dozen", "dozens": "dozen", "doz": "dozen",
    "piece": "piece", "pieces": "piece", "pc": "piece", "pcs": "piece",
    "unit": "unit", "units": "unit",
    "can": "can", "cans": "can", "tin": "can", "tins": "can",
    "jar": "jar", "jars": "jar",
    "tube": "tube", "tubes": "tube",
    "loaf": "loaf", "loaves": "loaf",
    "carton": "carton", "cartons": "carton",
    "bag": "bag", "bags": "bag",
    "bunch": "bunch", "bunches": "bunch",
    "bar": "bar", "bars": "bar",
    "roll": "roll", "rolls": "roll",
}

INVARIANT_UNITS = {"kg", "g", "mg", "lb", "ml"}
IRREGULAR_PLURALS = {"box": "boxes", "loaf": "loaves", "bunch": "bunches"}

# Conversion factors to a base unit (g for mass, ml for volume)
MASS = {"mg": 0.001, "g": 1.0, "kg": 1000.0, "lb": 453.592}
VOLUME = {"ml": 1.0, "litre": 1000.0}


def canonical_unit(token):
    """Return the canonical unit for a token, or None if it is not a unit."""
    return UNIT_ALIASES.get(str(token).lower().strip(".")) if token else None


def is_unit(token):
    return canonical_unit(token) is not None


def supported_units():
    return sorted(set(UNIT_ALIASES.values()))


def display_unit(unit, quantity=None):
    """Readable unit: 'litre' for 1, 'litres' for 2, 'kg' always."""
    if not unit:
        return ""
    if unit in INVARIANT_UNITS or quantity == 1:
        return unit
    return IRREGULAR_PLURALS.get(unit, unit + "s")


def convert(quantity, from_unit, to_unit):
    """Convert within the same family (mass or volume). None if incompatible."""
    if from_unit == to_unit:
        return quantity
    for family in (MASS, VOLUME):
        if from_unit in family and to_unit in family:
            return quantity * family[from_unit] / family[to_unit]
    return None


def format_number(value):
    if value is None:
        return ""
    value = round(float(value), 3)
    return str(int(value)) if value == int(value) else f"{value:g}"
