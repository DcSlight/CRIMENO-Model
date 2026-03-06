"""
Shared utilities and helpers for all workers
Reduces code duplication across workers
"""

import json
import zmq
import time
from typing import Dict, Any, Optional


class ZMQConnector:
    """Handles ZMQ socket creation and management"""
    
    @staticmethod
    def create_sub_socket(endpoint: str, topic: bytes = b"", timeout_ms: int = 1000) -> zmq.Socket:
        """Create and configure a SUB socket"""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(endpoint)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        return socket
    
    @staticmethod
    def create_pull_socket(endpoint: str) -> zmq.Socket:
        """Create and configure a PULL socket"""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(endpoint)
        return socket
    
    @staticmethod
    def create_push_socket(endpoint: str) -> zmq.Socket:
        """Create and configure a PUSH socket"""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket


class MessageHandler:
    """Handles message serialization/deserialization"""
    
    @staticmethod
    def encode_message(data: Dict[str, Any]) -> bytes:
        """Encode dict to JSON bytes"""
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    
    @staticmethod
    def decode_message(raw: bytes) -> Dict[str, Any]:
        """Decode JSON bytes to dict"""
        return json.loads(raw.decode("utf-8"))
    
    @staticmethod
    def add_timestamp(message: Dict[str, Any]) -> Dict[str, Any]:
        """Add generation timestamp to message"""
        message["generated_at_unix_ms"] = int(time.time() * 1000)
        return message


class Logger:
    """Simple logging utility"""
    
    @staticmethod
    def info(component: str, message: str):
        print(f"[{component}] ℹ️  {message}")
    
    @staticmethod
    def success(component: str, message: str):
        print(f"[{component}] ✅ {message}")
    
    @staticmethod
    def error(component: str, message: str):
        print(f"[{component}] ❌ {message}")
    
    @staticmethod
    def debug(component: str, message: str):
        print(f"[{component}] 🔍 {message}")


def safe_recv(socket: zmq.Socket, timeout: int = 100) -> Optional[Dict[str, Any]]:
    """Safely receive message with timeout and error handling"""
    try:
        msg = socket.recv(zmq.NOBLOCK)
        return MessageHandler.decode_message(msg)
    except zmq.Again:
        return None
    except Exception as e:
        Logger.error("ZMQ", f"Receive error: {e}")
        return None


def safe_send(socket: zmq.Socket, data: Dict[str, Any], component: str = "Worker") -> bool:
    """Safely send message with error handling"""
    try:
        msg = MessageHandler.encode_message(data)
        socket.send(msg)
        return True
    except Exception as e:
        Logger.error(component, f"Send failed: {e}")
        return False
