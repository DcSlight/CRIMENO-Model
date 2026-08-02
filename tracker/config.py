# Detection configuration for tracker_worker.
# Edit this file to tune thresholds or add/remove object classes —
# no changes to detection.py or tracker_worker.py needed.

# COCO classes (YOLO26 object model) relevant to robbery/surveillance.
# Everything else (chairs, TVs, plants...) is dropped before reaching the LLM.
# "person" is always kept regardless of this set.
ROBBERY_OBJECT_CLASSES = {
    "person", "backpack", "handbag", "suitcase",
    "knife", "cell phone", "bottle",
}
