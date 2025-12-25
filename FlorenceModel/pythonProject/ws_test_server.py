import asyncio
import websockets

async def handler(ws):
    print("✅ client connected")
    async for msg in ws:
        print("📦 got message:", msg[:120])

async def main():
    async with websockets.serve(handler, "127.0.0.1", 3000):
        print("WS test server on ws://127.0.0.1:3000/ws/tracker")
        await asyncio.Future()

asyncio.run(main())
