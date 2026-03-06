"""
Central configuration file for all workers
All ports, endpoints, and settings defined in one place
"""

# ============================================================
# ZeroMQ Configuration
# ============================================================
ZMQ_VIDEO_BROADCASTER_ENDPOINT = "tcp://127.0.0.1:5560"  # Video frames (PUB)
ZMQ_VIDEO_CMD_ENDPOINT = "tcp://127.0.0.1:5561"          # Video commands (REP)

# Worker communication ports
ZMQ_MESSAGE_BROKER_ENDPOINT = "tcp://127.0.0.1:5580"     # ALL workers → Message Broker (PUSH → PULL)
ZMQ_QWEN_INPUT_ENDPOINT = "tcp://127.0.0.1:5581"         # Message Broker → Qwen (PUSH → PULL)

# ============================================================
# WebSocket Configuration
# ============================================================
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 3000
WEBSOCKET_PATH = "/ws/broker"
WEBSOCKET_URL = f"ws://127.0.0.1:{WEBSOCKET_PORT}{WEBSOCKET_PATH}"

# ============================================================
# Model Configuration
# ============================================================
FLORENCE_MODEL = "florence-community/Florence-2-base"
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
YOLO_MODEL = "yolov8n.pt"

# ============================================================
# Worker Configuration
# ============================================================

# Florence Worker
FLORENCE_PROCESS_EVERY_N_FRAMES = 30
FLORENCE_OUTPUT_FILE = "analysis.jsonl"

# Tracker Worker
TRACKER_SEND_EVERY_N_FRAMES = 1
TRACKER_CONF_THRESHOLD = 0.35
TRACKER_IOU_THRESHOLD = 0.30
TRACKER_MAX_TRACK_AGE = 30
TRACKER_USE_MOTION_FALLBACK = True

# Qwen Anomaly Worker
QWEN_BASE_WINDOW_SIZE = 3
QWEN_MAX_QUEUE_SIZE = 30
QWEN_MAX_EVENT_HISTORY = 10
QWEN_CONTEXT_LOG_FILE = "qwen_context_log.txt"

# ============================================================
# Video Broadcaster
# ============================================================
VIDEO_RESIZE_WIDTH = 640
VIDEO_JPEG_QUALITY = 85

# ============================================================
# Message Types (for routing)
# ============================================================
MESSAGE_TYPE_FLORENCE_FRAME = "florence_frame"
MESSAGE_TYPE_TRACKER_FRAME = "tracker_frame"
MESSAGE_TYPE_QWEN_ANOMALY = "qwen_anomaly"
