# Detection configuration for tracker_worker.
# Edit this file to tune thresholds, add/remove object classes, or change
# open-vocab prompts — no changes to detection.py or tracker_worker.py needed.

# COCO classes (YOLO26 object model) relevant to robbery/surveillance.
# Everything else (chairs, TVs, plants...) is dropped before reaching the LLM.
# "person" is always kept regardless of this set.
ROBBERY_OBJECT_CLASSES = {
    "person", "backpack", "handbag", "suitcase",
    "knife", "cell phone", "bottle",
}

# Per-class confidence floors for the custom Suspicious_Activities nano model.
# Classes: Fighting, Man_With_Gun, Man_with_Knife, Theaf_Robbery.
# Weapon classes use lower floors for recall; precision is restored via
# person-overlap gating + temporal confirmation in the main loop.
SUSPICIOUS_CLASS_THRESHOLDS = {
    "Man_With_Gun":   0.45,
    "Man_with_Knife": 0.80,   # FP-prone on this model — keep strict
    "Theaf_Robbery":  0.55,
    "Fighting":       0.55,
}
DEFAULT_SUSPICIOUS_TH = 0.50

# Nano model classes that must overlap a detected person to be admitted.
# Kills floating "gun in mid-air" ghost detections.
WEAPON_SUSPICIOUS_CLASSES = {"Man_With_Gun", "Man_with_Knife"}

# Open-vocabulary APPEARANCE prompts for YOLOE-26.
# Concealment-focused only — "helmet" and "hooded person" removed because YOLOE
# fired them on construction hats / baseball caps constantly.
APPEARANCE_PROMPTS = [
    "hood", "ski mask", "balaclava", "face mask", "dark clothing",
]

# Open-vocabulary WEAPON prompts for YOLOE-26.
# Person-gated + temporally confirmed exactly like the nano weapon classes.
WEAPON_PROMPTS = ["gun", "pistol", "handgun", "rifle", "knife"]

# Everything YOLOE is asked to detect in a single pass.
YOLOE_PROMPTS = APPEARANCE_PROMPTS + WEAPON_PROMPTS

# All class names that count as "weapon" for person-gating + temporal confirmation,
# across both sources (nano model + open-vocab).
ALL_WEAPON_CLASSES = WEAPON_SUSPICIOUS_CLASSES | set(WEAPON_PROMPTS)
