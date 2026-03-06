#!/usr/bin/env python3
# message_broker.py
# Central message hub for all worker communication
# - Listens to ZMQ PULL socket (5580) from Florence, Tracker, Qwen
# - IMMEDIATELY forwards Florence/Tracker data to NestJS (no waiting!)
# - Also forwards Florence/Tracker data to Qwen for analysis
# - Routes messages to separate NestJS gateways based on message type:
#   - florence_frame → /ws/florence
#   - tracker_frame → /ws/tracker
#   - qwen_anomaly → /ws/qwen
# - Decouples models from frontend - models only know about ZMQ

import asyncio
import json
import zmq
import websockets
from typing import Optional
import logging
from config import ZMQ_MESSAGE_BROKER_ENDPOINT, ZMQ_QWEN_INPUT_ENDPOINT

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

ZMQ_PULL_ENDPOINT = ZMQ_MESSAGE_BROKER_ENDPOINT

# NestJS WebSocket endpoints (as client, connecting to NestJS gateways)
NESTJS_FLORENCE_WS = "ws://127.0.0.1:3000/ws/florence"
NESTJS_TRACKER_WS = "ws://127.0.0.1:3000/ws/tracker"
NESTJS_QWEN_WS = "ws://127.0.0.1:3000/ws/qwen"


class MessageBroker:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_socket.bind(ZMQ_PULL_ENDPOINT)
        
        # ZMQ socket to forward Florence/Tracker data to Qwen
        self.qwen_push_socket = self.zmq_context.socket(zmq.PUSH)
        self.qwen_push_socket.setsockopt(zmq.SNDHWM, 1000)
        self.qwen_push_socket.setsockopt(zmq.SNDTIMEO, 0)
        self.qwen_push_socket.connect(ZMQ_QWEN_INPUT_ENDPOINT)
        
        # WebSocket client connections to NestJS gateways
        self.florence_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.tracker_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.qwen_ws: Optional[websockets.WebSocketClientProtocol] = None
        
        logger.info(f"✅ ZMQ PULL bound on {ZMQ_PULL_ENDPOINT}")
        logger.info(f"✅ ZMQ PUSH to Qwen on {ZMQ_QWEN_INPUT_ENDPOINT}")

    def try_send_to_qwen(self, msg: bytes, msg_type: str, frame_idx) -> bool:
        """Non-blocking forward to Qwen. Never blocks broker loop."""
        try:
            self.qwen_push_socket.send(msg, flags=zmq.DONTWAIT)
            logger.info(f"➡️  Forwarded to Qwen | type={msg_type} | frame={frame_idx}")
            return True
        except zmq.Again:
            logger.warning(f"⚠️  Dropped for Qwen (backpressure) | type={msg_type} | frame={frame_idx}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed forwarding to Qwen | type={msg_type} | frame={frame_idx} | err={e}")
            return False
    
    async def connect_to_nestjs(self):
        """Connect to NestJS WebSocket gateways as a client."""
        # Connect to Florence gateway
        try:
            self.florence_ws = await websockets.connect(NESTJS_FLORENCE_WS)
            logger.info(f"✅ Connected to NestJS Florence gateway: {NESTJS_FLORENCE_WS}")
        except Exception as e:
            logger.error(f"⚠️  Failed to connect to Florence gateway: {e}")
        
        # Connect to Tracker gateway
        try:
            self.tracker_ws = await websockets.connect(NESTJS_TRACKER_WS)
            logger.info(f"✅ Connected to NestJS Tracker gateway: {NESTJS_TRACKER_WS}")
        except Exception as e:
            logger.error(f"⚠️  Failed to connect to Tracker gateway: {e}")
        
        # Connect to Qwen gateway
        try:
            self.qwen_ws = await websockets.connect(NESTJS_QWEN_WS)
            logger.info(f"✅ Connected to NestJS Qwen gateway: {NESTJS_QWEN_WS}")
        except Exception as e:
            logger.error(f"⚠️  Failed to connect to Qwen gateway: {e}")
    
    async def zmq_listener(self):
        """Listen for ZMQ messages from workers and route to appropriate NestJS gateway."""
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Non-blocking ZMQ receive with timeout
                msg = await loop.run_in_executor(None, self.zmq_socket.recv, zmq.NOBLOCK)
                msg_str = msg.decode("utf-8")
                
                try:
                    payload = json.loads(msg_str)
                except Exception as e:
                    logger.error(f"❌ Failed to parse JSON: {e}")
                    continue
                
                msg_type = payload.get("type", "unknown")
                
                # Route by message type
                if msg_type == "florence_frame":
                    frame_idx = payload.get("frame_index", "-")
                    logger.info(f"📨 Received florence_frame | frame={frame_idx}")
                    # Independent tunnels: do not await one destination before the other.
                    asyncio.create_task(self.send_to_florence(msg_str, frame_idx))
                    self.try_send_to_qwen(msg, msg_type, frame_idx)
                
                elif msg_type == "tracker_frame":
                    frame_idx = payload.get("frame_index", "-")
                    logger.info(f"📨 Received tracker_frame | frame={frame_idx}")
                    # Independent tunnels: do not await one destination before the other.
                    asyncio.create_task(self.send_to_tracker(msg_str, frame_idx))
                    self.try_send_to_qwen(msg, msg_type, frame_idx)
                
                elif msg_type == "qwen_anomaly":
                    frame_start = payload.get("frame_range", {}).get("start", "-")
                    logger.info(f"📨 Received qwen_anomaly | frame_start={frame_start}")
                    # Qwen output has its own tunnel to NestJS.
                    asyncio.create_task(self.send_to_qwen(msg_str, frame_start))
                
                else:
                    logger.warning(f"⚠️  Unknown message type: {msg_type}")
                
            except zmq.Again:
                # No message available, wait a bit
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"❌ ZMQ listener error: {e}")
                await asyncio.sleep(0.1)
    
    async def send_to_florence(self, message: str, frame_idx="-"):
        """Send message to NestJS Florence gateway."""
        if not self.florence_ws or self.florence_ws.closed:
            logger.warning("⚠️  Florence WS not connected")
            return
        
        try:
            await self.florence_ws.send(message)
            logger.info(f"✅ Sent to NestJS Florence | frame={frame_idx}")
        except Exception as e:
            logger.error(f"❌ Failed to send to Florence gateway: {e}")
            self.florence_ws = None
    
    async def send_to_tracker(self, message: str, frame_idx="-"):
        """Send message to NestJS Tracker gateway."""
        if not self.tracker_ws or self.tracker_ws.closed:
            logger.warning("⚠️  Tracker WS not connected")
            return
        
        try:
            await self.tracker_ws.send(message)
            logger.info(f"✅ Sent to NestJS Tracker | frame={frame_idx}")
        except Exception as e:
            logger.error(f"❌ Failed to send to Tracker gateway: {e}")
            self.tracker_ws = None
    
    async def send_to_qwen(self, message: str, frame_start="-"):
        """Send message to NestJS Qwen gateway."""
        if not self.qwen_ws or self.qwen_ws.closed:
            logger.warning("⚠️  Qwen WS not connected")
            return
        
        try:
            await self.qwen_ws.send(message)
            logger.info(f"✅ Sent to NestJS Qwen | frame_start={frame_start}")
        except Exception as e:
            logger.error(f"❌ Failed to send to Qwen gateway: {e}")
            self.qwen_ws = None
    
    async def run(self):
        """Main broker loop."""
        await self.connect_to_nestjs()
        await self.zmq_listener()


async def main():
    broker = MessageBroker()
    logger.info("🚀 Message Broker starting...")
    logger.info("=" * 60)
    logger.info("Architecture:")
    logger.info("  - Florence, Tracker, Qwen → ZMQ PULL 5580")
    logger.info("  - Message Broker (routes by message type):")
    logger.info("    - florence_frame → /ws/florence")
    logger.info("    - tracker_frame → /ws/tracker")
    logger.info("    - qwen_anomaly → /ws/qwen")
    logger.info("=" * 60)
    
    try:
        await broker.run()
    except KeyboardInterrupt:
        logger.info("\n✅ Message broker stopped by user.")
    finally:
        broker.zmq_socket.close()
        broker.zmq_context.term()


if __name__ == "__main__":
    asyncio.run(main())
