# new_cat = {
#     0: "void",
#     1: "base-cabinet",
#     2: "basket",
#     3: "bathtub",
#     4: "beanbag",
#     5: "bed",
#     6: "bench",
#     7: "bike",
#     8: "bin",
#     9: "blanket",
#     10: "blinds",
#     11: "book",
#     12: "bottle",
#     13: "bowl",
#     14: "box",
#     15: "cabinet",
#     16: "camera",
#     17: "candle",
#     18: "ceiling",
#     19: "chair",
#     20: "chopping-board",
#     21: "clock",
#     22: "clothing",
#     23: "comforter",
#     24: "cooktop",
#     25: "countertop",
#     26: "cup",
#     27: "curtain",
#     28: "cushion",
#     29: "desk",
#     30: "desk-organizer",
#     31: "door",
#     32: "faucet",
#     33: "floor", 
#     34: "handbag",
#     35: "handrail",
#     36: "indoor-plant",
#     37: "knife-block",
#     38: "lamp",
#     39: "major-appliance",
#     40: "mat",
#     41: "nightstand",
#     42: "panel",
#     43: "picture",
#     44: "pillar",
#     45: "pillow",
#     46: "pipe",
#     47: "plant-stand",
#     48: "plate",
#     49: "pot",
#     50: "rack",
#     51: "refrigerator",
#     52: "rug",
#     53: "scarf",
#     54: "sculpture",
#     55: "shelf",
#     56: "shoe",
#     57: "shower-stall",
#     58: "sink",
#     59: "small-appliance",
#     60: "sofa",
#     61: "stair",
#     62: "stool",
#     63: "switch",
#     64: "table",
#     65: "tablet",
#     66: "tissue-paper",
#     67: "toilet",
#     68: "tv-screen",
#     69: "tv-stand",
#     70: "umbrella",
#     71: "utensil-holder",
#     72: "vase",
#     73: "vent",
#     74: "wall",
#     75: "wall-cabinet",
#     76: "wall-plug",
#     77: "window",
# }


replica_cat = {
    0: 'void',
    1: 'backpack',
    2: 'base-cabinet',
    3: 'basket',
    4: 'bathtub',
    5: 'beam',
    6: 'beanbag',
    7: 'bed',
    8: 'bench',
    9: 'bike',
    10: 'bin',
    11: 'blanket',
    12: 'blinds',
    13: 'book',
    14: 'bottle',
    15: 'box',
    16: 'bowl',
    17: 'camera',
    18: 'cabinet',
    19: 'candle',
    20: 'chair',
    21: 'chopping-board',
    22: 'clock',
    23: 'cloth',
    24: 'clothing',
    25: 'coaster',
    26: 'comforter',
    27: 'computer-keyboard',
    28: 'cup', 
    29: 'cushion', 
    30: 'curtain',
    31: 'ceiling',
    32: 'cooktop',
    33: 'countertop',
    34: 'desk',
    35: 'desk-organizer',
    36: 'desktop-computer',
    37: 'door',
    38: 'exercise-ball',
    39: 'faucet',
    40: 'floor',
    41: 'handbag',
    42: 'hair-dryer', 
    43: 'handrail',
    44: 'indoor-plant',
    45: 'knife-block',
    46: 'kitchen-utensil',
    47: 'lamp',
    48: 'laptop',
    49: 'major-appliance',
    50: 'mat',
    51: 'microwave',
    52: 'monitor',
    53: 'mouse',
    54: 'nightstand',
    55: 'pan',
    56: 'panel',
    57: 'paper-towel',
    58: 'phone',
    59: 'picture',
    60: 'pillar',
    61: 'pillow',
    62: 'pipe',
    63: 'plant-stand',
    64: 'plate',
    65: 'pot',
    66: 'rack',
    67: 'refrigerator',
    68: 'remote-control', 
    69: 'scarf',
    70: 'sculpture',
    71: 'shelf',
    72: 'shoe',
    73: 'shower-stall',
    74: 'sink', 
    75: 'small-appliance', 
    76: 'sofa', 
    77: 'stair', 
    78: 'stool', 
    79: 'switch', 
    80: 'table', 
    81: 'table-runner', 
    82: 'tablet', 
    83: 'tissue-paper', 
    84: 'toilet', 
    85: 'toothbrush', 
    86: 'towel', 
    87: 'tv-screen', 
    88: 'tv-stand', 
    89: 'umbrella', 
    90: 'utensil-holder', 
    91: 'vase', 
    92: 'vent', 
    93: 'wall', 
    94: 'wall-cabinet', 
    95: 'wall-plug', 
    96: 'wardrobe', 
    97: 'window', 
    98: 'rug', 
    99: 'logo', 
    100: 'bag', 
    101: 'set-of-clothing',
    102: 'undefined'
    }


cat2id = {
    'backpack': 1, 
    'base-cabinet': 2, 
    'basket': 3, 
    'bathtub': 4, 
    'beam': 5, 
    'beanbag': 6, 
    'bed': 7, 
    'bench': 8, 
    'bike': 9, 
    'bin': 10, 
    'blanket': 11, 
    'blinds': 12, 
    'book': 13, 
    'bottle': 14, 
    'box': 15, 
    'bowl': 16, 
    'camera': 17, 
    'cabinet': 18, 
    'candle': 19, 
    'chair': 20, 
    'chopping-board': 21, 
    'clock': 22, 
    'cloth': 23, 
    'clothing': 24, 
    'coaster': 25, 
    'comforter': 26, 
    'computer-keyboard': 27, 
    'cup': 28, 
    'cushion': 29, 
    'curtain': 30, 
    'ceiling': 31, 
    'cooktop': 32, 
    'countertop': 33, 
    'desk': 34, 
    'desk-organizer': 35, 
    'desktop-computer': 36, 
    'door': 37, 
    'exercise-ball': 38, 
    'faucet': 39, 
    'floor': 40, 
    'handbag': 41, 
    'hair-dryer': 42, 
    'handrail': 43, 
    'indoor-plant': 44, 
    'knife-block': 45, 
    'kitchen-utensil': 46, 
    'lamp': 47, 
    'laptop': 48, 
    'major-appliance': 49, 
    'mat': 50, 
    'microwave': 51, 
    'monitor': 52, 
    'mouse': 53, 
    'nightstand': 54, 
    'pan': 55, 
    'panel': 56, 
    'paper-towel': 57, 
    'phone': 58, 
    'picture': 59, 
    'pillar': 60, 
    'pillow': 61, 
    'pipe': 62, 
    'plant-stand': 63, 
    'plate': 64, 
    'pot': 65, 
    'rack': 66, 
    'refrigerator': 67, 
    'remote-control': 68, 
    'scarf': 69, 
    'sculpture': 70, 
    'shelf': 71, 
    'shoe': 72, 
    'shower-stall': 73, 
    'sink': 74, 
    'small-appliance': 75, 
    'sofa': 76, 
    'stair': 77, 
    'stool': 78, 
    'switch': 79, 
    'table': 80, 
    'table-runner': 81, 
    'tablet': 82, 
    'tissue-paper': 83, 
    'toilet': 84, 
    'toothbrush': 85, 
    'towel': 86, 
    'tv-screen': 87, 
    'tv-stand': 88, 
    'umbrella': 89, 
    'utensil-holder': 90, 
    'vase': 91, 
    'vent': 92, 
    'wall': 93, 
    'wall-cabinet': 94,
    'wall-plug': 95, 
    'wardrobe': 96, 
    'window': 97, 
    'rug': 98, 
    'logo': 99, 
    'bag': 100, 
    'set-of-clothing': 101, 
    'undefined': 0,#-1, 
    'other-leaf': 0,#-1, 
    'anonymize_picture': 0,#-1, 
    'anonymize_text': 0,#-1}
    }