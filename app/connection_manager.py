import io
from PIL import Image
import numpy as np
from fastapi.websockets import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_binary(self, data: bytes):
        for connection in self.active_connections:
            await connection.send_bytes(data)

    async def broadcast_image(self, current_input_image: np.ndarray, max_size: int = 512, quality: int = 85, lossless: bool = True):
        """Broadcast image as PNG (lossless) or JPEG (lossy) with optional downsampling."""
        h, w = current_input_image.shape[:2]
        image_pil = Image.fromarray(current_input_image).convert("RGB")
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        if lossless:
            image_pil.save(buf, format="PNG", compress_level=1)
        else:
            image_pil.save(buf, format="JPEG", quality=quality)
        await self.broadcast_binary(buf.getvalue())
