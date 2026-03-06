#!/usr/bin/env python3
# message_broker.py
# Central message hub for all worker communication
# - Listens to ZMQ PULL socket (5580) from Florence, Tracker, Qwen
# - Broadcasts all messages via WebSocket (3000) to NestJS frontend
# - Decouples models from frontend - models only know about ZMQ

import asyncio
import json
import zmq
import websockets
from typing import Set, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

ZMQ_PULL_ENDPOINT = "tcp://127.0.0.1:5580"
WEBSOCKET_PORT = 3000
WEBSOCKET_PATH = "/ws/broker"


class MessageBroker:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_socket.bind(ZMQ_PULL_ENDPOINT)
        
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        logger.info(f"✅ ZMQ PULL bound on {ZMQ_PULL_ENDPOINT}")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection."""
        self.connected_clients.add(websocket)
        logger.info(f"🔗 WebSocket client connected (total: {len(self.connected_clients)})")
        
        try:
            async for message in websocket:
                # Just acknowledge clients can send messages
                pass
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"🔌 WebSocket client disconnected (total: {len(self.connected_clients)})")
    
    async def zmq_listener(self):
        """Listen for ZMQ messages from workers and broadcast to WebSocket clients."""
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Non-blocking ZMQ receive with timeout
                msg = await loop.run_in_executor(None, self.zmq_socket.recv, zmq.NOBLOCK)
                rec = json.loads(msg.decode("utf-8"))
                
                # Log incoming message
                msg_type = rec.get("type", "unknown")
                frame_idx = rec.get("frame_index", "-")
                logger.info(f"📨 Received {msg_type} | frame={frame_idx}")
                
                # Broadcast to all connected WebSocket clients
                await self.broadcast_to_clients(msg.decode("utf-8"))
                
            except zmq.Again:
                # No message available, wait a bit
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"❌ ZMQ listener error: {e}")
                await asyncio.sleep(0.1)
    
    async def broadcast_to_clients(self, message: str):
        """Broadcast message to all connected WebSocket clients."""
        if not self.connected_clients:
            return
        
        dead_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"❌ Failed to send to client: {e}")
                dead_clients.add(client)
        
        # Clean up dead connections
        for client in dead_clients:
            self.connected_clients.discard(client)
    
    async def run_websocket_server(self):
        """Start WebSocket server."""
        async with websockets.serve(self.handle_client, "0.0.0.0", WEBSOCKET_PORT):
            logger.info(f"🌐 WebSocket server listening on ws://0.0.0.0:{WEBSOCKET_PORT}{WEBSOCKET_PATH}")
            await asyncio.Future()  # Run forever
    
    async def start(self):
        """Start both ZMQ listener and WebSocket server."""
        await asyncio.gather(
            self.zmq_listener(),
            self.run_websocket_server()
        )


async def main():
    broker = MessageBroker()
    logger.info("🚀 Message Broker starting...")
    logger.info("=" * 60)
    logger.info("Architecture:")
    logger.info("  - Florence, Tracker, Qwen → ZMQ PULL 5580")
    logger.info("  - Message Broker → WebSocket 3000 → NestJS")
    logger.info("=" * 60)
    
    try:
        await broker.start()
    except KeyboardInterrupt:
        logger.info("\n[INFO] Message broker stopped by user.")
    finally:
        broker.zmq_socket.close()
        broker.zmq_context.term()


if __name__ == "__main__":
    asyncio.run(main())
