"""Configured category dictionary (the ONLY source of category knowledge).

Keys are singular, lower-case product names. Multi-word entries are allowed.
Anything not found here (or matched by the guarded similarity step) is
reported as "Other" - the classifier never guesses.
"""

CATEGORIES = [
    "Fruits", "Vegetables", "Dairy", "Grains", "Bakery", "Meat", "Seafood",
    "Beverages", "Snacks", "Spices", "Condiments", "Frozen Foods",
    "Personal Care", "Household", "Cleaning", "Stationery", "Other",
]

CATEGORY_KEYWORDS = {
    "Fruits": [
        "apple", "banana", "orange", "mango", "grape", "watermelon", "melon",
        "papaya", "pineapple", "pear", "peach", "plum", "cherry", "strawberry",
        "blueberry", "pomegranate", "guava", "kiwi", "lemon", "lime", "coconut",
        "apricot", "fig", "date", "lychee", "raspberry", "tangerine",
    ],
    "Vegetables": [
        "tomato", "potato", "onion", "carrot", "spinach", "cabbage",
        "cauliflower", "broccoli", "cucumber", "capsicum", "pepper", "garlic",
        "ginger", "lettuce", "pea", "bean", "brinjal", "eggplant", "okra",
        "pumpkin", "beetroot", "radish", "mushroom", "corn", "zucchini",
        "coriander", "mint", "celery", "turnip", "vegetable",
    ],
    "Dairy": [
        "milk", "cheese", "butter", "yogurt", "yoghurt", "curd", "cream",
        "paneer", "ghee", "egg", "buttermilk", "ice cream", "whey",
    ],
    "Grains": [
        "rice", "wheat", "flour", "oats", "oat", "cereal", "pasta", "noodle",
        "macaroni", "spaghetti", "barley", "quinoa", "semolina", "lentil",
        "dal", "millet", "cornflakes", "muesli", "maize", "chickpea",
    ],
    "Bakery": [
        "bread", "cake", "bun", "croissant", "pastry", "muffin", "bagel",
        "roll", "toast", "cookie", "donut", "doughnut", "pav", "baguette",
    ],
    "Meat": [
        "chicken", "mutton", "beef", "pork", "lamb", "turkey", "sausage",
        "bacon", "ham", "salami", "meat", "steak", "mince",
    ],
    "Seafood": [
        "fish", "prawn", "shrimp", "crab", "lobster", "salmon", "tuna",
        "squid", "oyster", "sardine", "seafood", "mackerel",
    ],
    "Beverages": [
        "juice", "water", "tea", "coffee", "soda", "cola", "lemonade",
        "smoothie", "beer", "wine", "energy drink", "squash", "drink",
        "coke", "pepsi",
    ],
    "Snacks": [
        "biscuit", "chip", "chocolate", "candy", "popcorn", "cracker",
        "namkeen", "nut", "almond", "cashew", "peanut", "walnut", "raisin",
        "snack", "wafer", "pretzel", "granola bar", "bhujia",
    ],
    "Spices": [
        "salt", "turmeric", "cumin", "cinnamon", "clove", "cardamom",
        "masala", "chilli", "chili", "oregano", "spice", "saffron", "paprika",
        "nutmeg", "fennel", "mustard seed", "black pepper", "coriander powder",
    ],
    "Condiments": [
        "sugar", "oil", "vinegar", "ketchup", "sauce", "mayonnaise", "honey",
        "jam", "pickle", "mustard", "syrup", "olive oil", "cooking oil",
        "soy sauce", "spread", "jaggery", "peanut butter",
    ],
    "Frozen Foods": [
        "frozen peas", "frozen pizza", "pizza", "nugget", "frozen vegetable",
        "frozen food", "french fries", "fries", "frozen meal", "popsicle",
    ],
    "Personal Care": [
        "soap", "shampoo", "toothpaste", "toothbrush", "conditioner",
        "deodorant", "lotion", "razor", "moisturizer", "sunscreen", "perfume",
        "face wash", "body wash", "mouthwash", "comb", "sanitary pad",
        "hair oil", "cotton",
    ],
    "Household": [
        "tissue", "toilet paper", "foil", "trash bag", "garbage bag", "candle",
        "battery", "bulb", "matchbox", "napkin", "paper towel", "plastic wrap",
        "light bulb", "lighter", "hanger",
    ],
    "Cleaning": [
        "detergent", "dishwash", "dishwasher", "floor cleaner", "bleach",
        "sponge", "mop", "broom", "disinfectant", "cleaner", "phenyl",
        "toilet cleaner", "glass cleaner", "scrubber", "dish soap",
        "washing powder", "fabric softener",
    ],
    "Stationery": [
        "pen", "pencil", "notebook", "eraser", "marker", "stapler", "glue",
        "ruler", "highlighter", "sharpener", "paper", "folder", "envelope",
        "tape", "scissors", "diary",
    ],
}
