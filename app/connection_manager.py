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
            # Resize using numpy/cv2 would be faster, but PIL is simpler
            image_pil = Image.fromarray(current_input_image).convert("RGB")
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.NEAREST)
            current_input_image = np.array(image_pil)

        # Send raw RGB data with dimensions header
        # Format: 4 bytes width + 4 bytes height + raw RGB data
        h, w = current_input_image.shape[:2]
        header = w.to_bytes(4, 'little') + h.to_bytes(4, 'little')
        img_bytes = header + current_input_image.tobytes()

        await self.broadcast_binary(img_bytes)
