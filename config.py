"""
PreFlight AI - Configuration Module
====================================
Central configuration for corruption parameters, agent profiles, and scoring.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =============================================================================
# IMAGE NET CLASS LABELS (Full 1000 classes embedded for offline use)
# =============================================================================
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "American egret", "bittern", "crane", "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
    "Afghan hound", "basset", "beagle", "bloodhound", "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound", "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
    "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
    "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
    "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
    "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
    "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog",
    "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
    "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
    "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
    "white wolf", "red wolf", "coyote", "dingo", "dhole", "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
    "ant", "grasshopper", "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly", "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
    "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn",
    "ibex", "hartebeest", "impala", "gazelle", "Arabian camel", "llama", "weasel", "mink", "polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
    "guenon", "patas", "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
    "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri", "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibian", "analog clock",
    "apiary", "apron", "ashcan", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint", "Band Aid",
    "banjo", "bannister", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "barrow", "baseball",
    "basketball", "bassinet", "bassoon", "bathing cap", "bath towel", "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
    "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini", "binder", "binoculars", "birdhouse", "boathouse",
    "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop", "bottlecap", "bow", "bow tie", "brass", "brassiere",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron",
    "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
    "cash machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "cellular telephone", "chain", "chainlink fence",
    "chain mail", "chain saw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "cinema", "cleaver",
    "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery",
    "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane", "crash helmet", "crate",
    "crib", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishrag", "dishwasher", "disk brake", "dock", "dogsled", "dome", "doormat",
    "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
    "espresso maker", "face powder", "feather boa", "file", "fireboat", "fire engine", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster", "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gasmask", "gas pump", "goblet", "go-kart", "golf ball", "golfcart", "gondola", "gong", "gown", "grand piano",
    "greenhouse", "grille", "grocery store", "guillotine", "hair slide", "hair spray", "half track", "hammer", "hamper", "hand blower",
    "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass", "iPod", "iron", "jack-o'-lantern", "jean", "jeep",
    "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade",
    "laptop", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "liner", "lipstick",
    "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic compass", "mailbag", "mailbox", "maillot", "maillot",
    "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine chest", "megalith",
    "microphone", "microwave", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito net",
    "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace",
    "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
    "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush", "pajama", "palace", "panpipe",
    "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "pay-phone", "pedestal", "pencil box",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick", "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane", "planetarium", "plastic bag", "plate rack",
    "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "puck", "punching bag", "purse", "quill",
    "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera",
    "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
    "running shoe", "safe", "safety pin", "saltshaker", "sandal", "sarong", "sax", "scabbard", "scale", "school bus",
    "schooner", "scoreboard", "screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
    "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
    "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web", "spindle", "sports car", "spotlight", "stage",
    "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer", "streetcar",
    "stretcher", "studio couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
    "swab", "sweatshirt", "swimming trunks", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy", "television", "tennis ball", "thatch", "theater curtain", "thimble", "thresher", "throne", "tile roof", "toaster",
    "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toyshop", "tractor", "trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
    "unicycle", "upright", "vacuum", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin",
    "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "warplane", "washbasin", "washer", "water bottle", "water jug",
    "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok",
    "wooden spoon", "wool", "worm fence", "wreck", "yawl", "yurt", "web site", "comic book", "crossword puzzle", "street sign",
    "traffic light", "book jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
    "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
    "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
    "chocolate sauce", "dough", "meat loaf", "pizza", "potpie", "burrito", "red wine", "espresso", "cup", "eggnog",
    "alp", "bubble", "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley",
    "volcano", "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
    "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"
]

# =============================================================================
# CORRUPTION CONFIGURATION
# =============================================================================

@dataclass
class CorruptionConfig:
    """Configuration for a single corruption type."""
    name: str
    category: str
    description: str
    severity_range: Tuple[float, float]  # (min, max) for severity 1-5
    agent_tags: List[str]  # Tags for agent-based selection
    params: Dict  # Additional parameters

# Corruption definitions with graduated severity
CORRUPTIONS = {
    # Environmental Degradations
    "gaussian_noise": CorruptionConfig(
        name="Gaussian Noise",
        category="Environmental",
        description="Simulates sensor noise and low-light conditions",
        severity_range=(0.04, 0.20),
        agent_tags=["low_light", "night", "sensor", "indoor", "noisy"],
        params={"color": False}
    ),
    "motion_blur": CorruptionConfig(
        name="Motion Blur",
        category="Environmental",
        description="Simulates camera shake or moving objects",
        severity_range=(7, 23),
        agent_tags=["moving", "vehicle", "handheld", "action", "sports"],
        params={"direction": "random"}
    ),
    "fog": CorruptionConfig(
        name="Fog",
        category="Environmental",
        description="Atmospheric obstruction and haze",
        severity_range=(0.15, 0.75),
        agent_tags=["outdoor", "weather", "distance", "aerial", "marine"],
        params={"turbidity": 0.5}
    ),
    "rain": CorruptionConfig(
        name="Rain Streaks",
        category="Environmental",
        description="Simulates rain and water droplets on lens",
        severity_range=(1.0, 3.0),
        agent_tags=["outdoor", "weather", "driving", "marine", "security"],
        params={"drop_color": (180, 180, 180)}
    ),
    "brightness": CorruptionConfig(
        name="Brightness Variation",
        category="Environmental",
        description="Lighting changes and exposure variations",
        severity_range=(0.15, 0.75),
        agent_tags=["indoor", "outdoor", "lighting", "daynight", "exposure"],
        params={"direction": "both"}
    ),

    # Digital Artifacts
    "jpeg_compression": CorruptionConfig(
        name="JPEG Compression",
        category="Digital",
        description="Lossy compression artifacts from transmission",
        severity_range=(95, 20),
        agent_tags=["streaming", "mobile", "bandwidth", "storage", "transmission"],
        params={"subsampling": "4:2:0"}
    ),
    "pixelation": CorruptionConfig(
        name="Pixelation",
        category="Digital",
        description="Low resolution and upscaling artifacts",
        severity_range=(4, 12),
        agent_tags=["low_res", "thumbnail", "zoom", "legacy", "cctv"],
        params={"mode": "block"}
    ),
    "gaussian_blur": CorruptionConfig(
        name="Gaussian Blur",
        category="Digital",
        description="Defocus and lens blur effects",
        severity_range=(2, 6),
        agent_tags=["focus", "lens", "portrait", "depth", "optical"],
        params={"kernel_type": "gaussian"}
    ),

    # Geometric Transformations
    "occlusion": CorruptionConfig(
        name="Occlusion",
        category="Geometric",
        description="Partial obstruction of objects",
        severity_range=(0.10, 0.25),
        agent_tags=["crowded", "cluttered", "partial", "overlapping", "occluded"],
        params={"mode": "random_patch"}
    ),
    "rotation": CorruptionConfig(
        name="Rotation",
        category="Geometric",
        description="Perspective and orientation variations",
        severity_range=(7, 35),
        agent_tags=["tilted", "angle", "perspective", "handheld", "mounted"],
        params={"crop": True}
    ),

    # Adversarial Attacks (ML Component)
    "fgsm": CorruptionConfig(
        name="FGSM Attack",
        category="Adversarial",
        description="Fast Gradient Sign Method - gradient-based perturbation",
        severity_range=(0.01, 0.10),
        agent_tags=["security", "attack", "gradient", "white_box", "fast"],
        params={"targeted": False}
    ),
    "pgd": CorruptionConfig(
        name="PGD Attack",
        category="Adversarial",
        description="Projected Gradient Descent - iterative adversarial attack",
        severity_range=(0.01, 0.05),
        agent_tags=["security", "attack", "gradient", "white_box", "iterative"],
        params={"iterations": 10, "step_size": 0.01}
    ),
}

# =============================================================================
# AGENT PROFILES
# =============================================================================

@dataclass
class AgentProfile:
    """Intelligent agent that selects corruptions based on image analysis."""
    name: str
    description: str
    icon: str
    trigger_conditions: List[str]
    corruption_weights: Dict[str, float]
    reasoning: str

AGENTS = {
    "night_vision": AgentProfile(
        name="Night Vision Agent",
        description="Specializes in low-light and nighttime scenarios",
        icon="🌙",
        trigger_conditions=["dark", "night", "low_light", "indoor_dark"],
        corruption_weights={
            "gaussian_noise": 1.5,
            "brightness": 1.3,
            "motion_blur": 1.0,
            "fog": 0.5,
            "jpeg_compression": 0.8,
        },
        reasoning="Low-light images suffer from sensor noise and exposure issues. Prioritizing noise and brightness corruptions."
    ),
    "action_cam": AgentProfile(
        name="Action Camera Agent",
        description="Handles fast-moving subjects and dynamic scenes",
        icon="🏃",
        trigger_conditions=["motion", "sports", "vehicle", "fast", "dynamic"],
        corruption_weights={
            "motion_blur": 1.5,
            "rotation": 1.2,
            "gaussian_noise": 0.8,
            "brightness": 0.7,
        },
        reasoning="Motion-heavy scenarios require blur and orientation testing. Prioritizing motion blur and rotation."
    ),
    "surveillance": AgentProfile(
        name="Surveillance Agent",
        description="Optimizes for CCTV and security camera deployments",
        icon="📹",
        trigger_conditions=["cctv", "security", "monitoring", "indoor", "fixed"],
        corruption_weights={
            "jpeg_compression": 1.4,
            "pixelation": 1.3,
            "gaussian_noise": 1.1,
            "occlusion": 1.0,
            "brightness": 0.9,
        },
        reasoning="Surveillance systems use heavy compression and low resolution. Testing compression and pixelation robustness."
    ),
    "autonomous": AgentProfile(
        name="Autonomous Vehicle Agent",
        description="Tests for self-driving and automotive vision systems",
        icon="🚗",
        trigger_conditions=["outdoor", "road", "vehicle", "weather", "driving"],
        corruption_weights={
            "fog": 1.5,
            "motion_blur": 1.3,
            "brightness": 1.2,
            "gaussian_noise": 0.9,
            "occlusion": 1.1,
        },
        reasoning="Autonomous vehicles face weather, motion, and lighting challenges. Prioritizing fog, blur, and brightness."
    ),
    "mobile_app": AgentProfile(
        name="Mobile App Agent",
        description="Simulates mobile camera and social media uploads",
        icon="📱",
        trigger_conditions=["mobile", "social", "casual", "selfie", "upload"],
        corruption_weights={
            "jpeg_compression": 1.4,
            "brightness": 1.2,
            "rotation": 1.1,
            "gaussian_blur": 0.9,
            "pixelation": 1.0,
        },
        reasoning="Mobile uploads have compression, lighting, and orientation issues. Testing compression and brightness."
    ),
    "security_ai": AgentProfile(
        name="Security AI Agent",
        description="Tests adversarial robustness for security-critical systems",
        icon="🔒",
        trigger_conditions=["security", "critical", "adversarial", "attack", "robust"],
        corruption_weights={
            "fgsm": 1.5,
            "pgd": 1.5,
            "gaussian_noise": 1.0,
            "jpeg_compression": 0.8,
        },
        reasoning="Security-critical systems must resist adversarial attacks. Prioritizing FGSM and PGD testing."
    ),
    "medical": AgentProfile(
        name="Medical Imaging Agent",
        description="Validates clinical and diagnostic imaging systems",
        icon="🏥",
        trigger_conditions=["medical", "clinical", "diagnostic", "xray", "scan"],
        corruption_weights={
            "gaussian_noise": 1.4,
            "brightness": 1.3,
            "jpeg_compression": 1.2,
            "gaussian_blur": 1.0,
        },
        reasoning="Medical images require noise and contrast stability. Testing noise, brightness, and compression."
    ),
    "general": AgentProfile(
        name="General Purpose Agent",
        description="Balanced testing across all corruption categories",
        icon="🔬",
        trigger_conditions=["general", "balanced", "default", "mixed"],
        corruption_weights={
            "gaussian_noise": 1.0,
            "motion_blur": 1.0,
            "brightness": 1.0,
            "jpeg_compression": 1.0,
            "occlusion": 1.0,
            "rotation": 1.0,
            "fog": 1.0,
            "gaussian_blur": 1.0,
            "pixelation": 1.0,
            "fgsm": 0.8,
            "pgd": 0.8,
        },
        reasoning="Balanced approach testing all corruption types equally for general deployment scenarios."
    ),
}

# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

RISK_THRESHOLDS = {
    "low": (0, 30),
    "medium": (31, 60),
    "high": (61, 100),
}

RISK_COLORS = {
    "low": "#2ecc71",
    "medium": "#f39c12",
    "high": "#e74c3c",
}

SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# Patent-aligned core set: 6 corruptions x 3 severities = 18 variants
CORE_CORRUPTION_KEYS = ["gaussian_noise", "motion_blur", "brightness", "occlusion", "jpeg_compression", "rain"]
