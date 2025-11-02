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

    async def broadcast_image(self, current_input_image: np.ndarray, max_size: int = 512):
        """Broadcast image with optional downsampling for performance."""
        # Downsample if image is larger than max_size
        h, w = current_input_image.shape[:2]
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_pil = Image.fromarray(current_input_image).convert("RGB")
            # Use BILINEAR for faster resizing (LANCZOS is slower but higher quality)
            image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)
        else:
            image_pil = Image.fromarray(current_input_image).convert("RGB")

        with io.BytesIO() as buffered:
            # Adjust quality based on image size for faster encoding
            quality = 75 if h > max_size or w > max_size else 85
            image_pil.save(buffered, format="JPEG", quality=quality, optimize=False)
            img_bytes = buffered.getvalue()
        await self.broadcast_binary(img_bytes)