# ========================
# Standard Library Imports
# ========================
import os
import asyncio

# ========================
# Third-Party Imports
# ========================
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ========================
# Local Module Imports
# ========================
from app.connection_manager import ConnectionManager
from app.state_handler import StateHandler
from app.ca_handler import CaHandler
from app.tools.brushes import delete_brush, paint_brush

# ========================
# Device Configuration
# ========================
def get_device_preference():
    """Get device preference from environment variable."""
    device = os.environ.get("NCA_DEVICE", None)

    # If device is set to cpu, hide CUDA devices
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("NCA_DEVICE=cpu: Hiding CUDA devices")

    if device:
        print(f"Device preference from NCA_DEVICE: {device}")
    else:
        print("No device preference set (will auto-detect)")

    return device

PREFERRED_DEVICE = get_device_preference()

# ========================
# Global Constants & Configurations
# ========================
TRAINLOG_ROOT = "./train_log"
CANVAS_SIZE = 512

# ========================
# FastAPI App Initialization
# ========================
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files (e.g., scripts)
app.mount("/static/js", StaticFiles(directory="app/scripts"), name="static")

# Configure CORS middleware (adjust allow_origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Asyncio Locks & Events
# ========================
image_lock = asyncio.Lock()
x_lock = asyncio.Lock()
random_changes_event = asyncio.Event()

# ========================
# Initialize CA and State Handlers
# ========================
log_path = "./train_log/emoji_conv3x3"  # Adjust as needed
current_model_folder = "emoji_conv3x3"  # Track the current model folder
ca_handler = CaHandler(device_preference=PREFERRED_DEVICE)
config = ca_handler.load_model(log_path)
print(f"✓ Model loaded on device: {config.DEVICE}")
IMG_SIZE, COLOR_DIM, COND_DIM, x_tensor, current_input_image, cond = ca_handler.get_initial_data()

state_handler = StateHandler(config, IMG_SIZE, COLOR_DIM, COND_DIM)
state_handler.set_state(current_input_image[:, :, :3], x_tensor, cond)

# Manager for handling WebSocket connections
manager = ConnectionManager()


# ========================
# Utility Functions
# ========================
def _to_cpu_detached(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU without tracking gradients."""
    return tensor.detach().cpu()


def convert_to_img(x_tensor: torch.Tensor, config, color_dim: int):
    """
    Convert the CA tensor state to an RGB image using the dataset's state_to_img method.
    """
    return ca_handler.get_dataset().state_to_img(x_tensor, color_dim)


async def reset_seed_and_condition():
    """
    Reset the CA seed and condition, update the state, and broadcast the new image.
    """
    _, _, _, x_tensor, current_input_image, cond = ca_handler.get_initial_data()
    with state_handler.get_lock():
        state_handler.set_state(current_input_image[:, :, :3], x_tensor, cond)
        state_handler.set_seed_tensor(x_tensor.clone().detach())
    await manager.broadcast_image(state_handler.get_img_canvas())


async def apply_ca_changes():
    """
    Continuously apply CA changes at intervals defined by the current speed.
    Implements adaptive frame skipping for better performance with large images.
    """
    frame_counter = 0
    import time
    last_log_time = time.time()
    frame_times = []
    timing_breakdown = {'forward': [], 'convert': [], 'broadcast': []}

    while random_changes_event.is_set():
        loop_start = time.time()
        # Only sleep if speed setting requires it (don't sleep for high FPS)
        sleep_time = state_handler.get_speed()
        if sleep_time > 0.001:  # Only sleep if > 1ms
            await asyncio.sleep(sleep_time)

        with state_handler.get_lock():
            x_tensor = state_handler.get_img_tensor().clone().detach()
            cond_tensor = state_handler.get_condition_tensor()
            cond_tensor = cond_tensor.clone().detach() if cond_tensor is not None else None
            config = state_handler.get_config()
            color_dim = state_handler.get_color_dim()

        # Adaptive broadcast interval based on image size and speed
        img_size = x_tensor.shape[-1] * x_tensor.shape[-2]
        speed = state_handler.get_speed()

        # Aggressive frame skipping to compensate for slow encoding
        # Skip more frames to maintain responsiveness
        if img_size > 512 * 512:
            broadcast_interval = 4  # Every 4th frame for very large images
        else:
            broadcast_interval = 2  # Every 2nd frame for 512x512 to compensate for encoding time

        # Time the forward pass
        t0 = time.time()
        x_tensor_gpu, x_latent_gpu = await asyncio.to_thread(
            ca_handler.forward, x_tensor, cond_tensor
        )
        # For latent training, store the latent for next iteration
        # but keep the decoded tensor for display
        if config.LATENT_TRAINING.ENABLED:
            x_latent_cpu = await asyncio.to_thread(_to_cpu_detached, x_latent_gpu)
            x_display_cpu = await asyncio.to_thread(_to_cpu_detached, x_tensor_gpu)
        else:
            x_display_cpu = await asyncio.to_thread(_to_cpu_detached, x_tensor_gpu)
            x_latent_cpu = None
        t1 = time.time()
        timing_breakdown['forward'].append((t1 - t0) * 1000)

        # Time the image conversion
        img = await asyncio.to_thread(convert_to_img, x_display_cpu, config, color_dim)
        t2 = time.time()
        timing_breakdown['convert'].append((t2 - t1) * 1000)

        with state_handler.get_lock():
            if config.LATENT_TRAINING.ENABLED:
                state_handler.set_img_tensor(x_latent_cpu)
            else:
                state_handler.set_img_tensor(x_display_cpu)

            state_handler.set_img_canvas(img)
            img_for_broadcast = img.copy()

        # Only broadcast every Nth frame to reduce network overhead
        frame_counter += 1
        if frame_counter >= broadcast_interval:
            await manager.broadcast_image(img_for_broadcast)
            t3 = time.time()
            timing_breakdown['broadcast'].append((t3 - t2) * 1000)
            frame_counter = 0
        else:
            timing_breakdown['broadcast'].append(0)  # Frame skipped

        # Performance monitoring
        loop_time = time.time() - loop_start
        frame_times.append(loop_time)
        if time.time() - last_log_time > 5.0:  # Log every 5 seconds
            avg_time = sum(frame_times) / len(frame_times)
            actual_fps = 1.0 / avg_time if avg_time > 0 else 0

            # Calculate average timings
            avg_forward = sum(timing_breakdown['forward']) / len(timing_breakdown['forward'])
            avg_convert = sum(timing_breakdown['convert']) / len(timing_breakdown['convert'])
            broadcast_times = [t for t in timing_breakdown['broadcast'] if t > 0]
            avg_broadcast = sum(broadcast_times) / len(broadcast_times) if broadcast_times else 0

            device_info = config.DEVICE
            broadcast_fps = len(broadcast_times) / 5.0  # broadcasts per second
            print(f"Performance: {actual_fps:.1f} FPS (total: {avg_time*1000:.1f}ms) | "
                  f"Forward: {avg_forward:.1f}ms | Convert: {avg_convert:.1f}ms | "
                  f"Broadcast: {avg_broadcast:.1f}ms ({broadcast_fps:.1f} fps) | Device: {device_info} | Image: {x_tensor.shape[-2]}x{x_tensor.shape[-1]}")

            frame_times = []
            timing_breakdown = {'forward': [], 'convert': [], 'broadcast': []}
            last_log_time = time.time()


def visualize_feature_space_max_activation(feature_tensor):
    """
    Visualize a 16x64x64 feature space as a 3x64x64 RGB image using max-activation.
    
    Args:
        feature_tensor (torch.Tensor): Feature tensor of shape (16, 64, 64)
        
    Returns:
        torch.Tensor: RGB image of shape (3, 64, 64)
    """
    max_activations, indices = torch.max(torch.abs(feature_tensor), dim=0)  # (64, 64)
    # Map feature index to RGB channels using modulo arithmetic
    r = (indices % 3 == 0).float() * max_activations
    g = (indices % 3 == 1).float() * max_activations
    b = (indices % 3 == 2).float() * max_activations
    output = torch.stack([r, g, b], dim=0)
    output = (output - output.min()) / (output.max() - output.min())
    return output


def get_trainlog_folders():
    folders = []
    for root, dirs, files in os.walk(TRAINLOG_ROOT):
        if "ca_final.pt" in files:
            # Return relative path starting with "train_log"
            rel_path = os.path.join(os.path.relpath(root, TRAINLOG_ROOT))
            folders.append(rel_path)
    return sorted(folders, reverse=True)



def resize_canvas_centered(img_canvas: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resize a numpy image canvas to a new size by centered cropping or padding.
    """
    old_height, old_width = img_canvas.shape[:2]
    new_canvas = np.zeros((new_height, new_width, img_canvas.shape[2]), dtype=img_canvas.dtype)

    # Determine vertical cropping/padding
    if new_height >= old_height:
        dest_y = (new_height - old_height) // 2
        src_y = 0
        copy_height = old_height
    else:
        dest_y = 0
        src_y = (old_height - new_height) // 2
        copy_height = new_height

    # Determine horizontal cropping/padding
    if new_width >= old_width:
        dest_x = (new_width - old_width) // 2
        src_x = 0
        copy_width = old_width
    else:
        dest_x = 0
        src_x = (old_width - new_width) // 2
        copy_width = new_width

    new_canvas[dest_y:dest_y+copy_height, dest_x:dest_x+copy_width] = \
        img_canvas[src_y:src_y+copy_height, src_x:src_x+copy_width]
    return new_canvas


def resize_tensor_centered(img_tensor: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    """
    Resize a torch image tensor to a new size by centered cropping or padding.
    """
    batch, channels, old_height, old_width = img_tensor.shape
    new_tensor = torch.zeros((batch, channels, new_height, new_width), dtype=img_tensor.dtype, device=img_tensor.device)
    
    # Determine vertical cropping/padding
    if new_height >= old_height:
        dest_y = (new_height - old_height) // 2
        src_y = 0
        copy_height = old_height
    else:
        dest_y = 0
        src_y = (old_height - new_height) // 2
        copy_height = new_height

    # Determine horizontal cropping/padding
    if new_width >= old_width:
        dest_x = (new_width - old_width) // 2
        src_x = 0
        copy_width = old_width
    else:
        dest_x = 0
        src_x = (old_width - new_width) // 2
        copy_width = new_width

    new_tensor[:, :, dest_y:dest_y+copy_height, dest_x:dest_x+copy_width] = \
        img_tensor[:, :, src_y:src_y+copy_height, src_x:src_x+copy_width]
    return new_tensor


# ========================
# FastAPI Routes
# ========================

# --- Main Page & WebSocket ---

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    """
    Render the base template with available trainlog folders.
    """
    folders = get_trainlog_folders()
    return templates.TemplateResponse(request, "base.html", {"folders": folders})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Return an empty favicon response.
    """
    return Response(content="", media_type="image/x-icon")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle WebSocket connections for live image updates.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- Model & Image Manipulation Routes ---

@app.post("/erase")
async def erase(action: dict):
    """
    Erase part of the image using a brush tool.
    """
    with state_handler.get_lock():
        img_canvas = state_handler.get_img_canvas()
        img_tensor = state_handler.get_img_tensor().squeeze(0)
        img_canvas, img_tensor = delete_brush(
            img_canvas, img_tensor, action, img_canvas.shape[0]
        )
        state_handler.set_img_canvas(img_canvas)
        state_handler.set_img_tensor(img_tensor.unsqueeze(0))
    return JSONResponse(content={"status": "Erased"})


@app.post("/set_seed_dot")
async def set_seed_dot(action: dict):
    """
    Set a seed dot on the image based on provided coordinates.
    """
    with state_handler.get_lock():
        img_canvas = state_handler.get_img_canvas()
        img_tensor = state_handler.get_img_tensor().squeeze(0)
        x = int(action["x"])
        y = int(action["y"])
        # Rescale coordinates to match the canvas size
        x = int(x * img_canvas.shape[0] / CANVAS_SIZE)
        y = int(y * img_canvas.shape[0] / CANVAS_SIZE)
        img_tensor[:, y, x] = 1.0
        state_handler.set_img_tensor(img_tensor.unsqueeze(0))
    return JSONResponse(content={"status": "Seed dot set"})


@app.post("/reset_image")
async def reset_image():
    """
    Reset the image to its initial state.
    """
    await reset_seed_and_condition()
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Image cleared"})


@app.post("/clear_image")
async def clear_image():
    """
    Clear the image canvas.
    """
    with state_handler.get_lock():
        img_canvas = state_handler.get_img_canvas() * 0
        state_handler.set_img_canvas(img_canvas)
        img_tensor = state_handler.get_img_tensor()
        img_tensor[0:4] = img_tensor[0:4] * 0
        state_handler.set_img_tensor(img_tensor)
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Image cleared"})


@app.post("/stop_changes")
async def stop_changes():
    """
    Stop applying random CA changes.
    """
    random_changes_event.clear()
    return JSONResponse(content={"status": "Random changes stopped"})


@app.post("/start_changes")
async def start_changes():
    """
    Start applying random CA changes.
    """
    random_changes_event.set()
    asyncio.create_task(apply_ca_changes())
    return JSONResponse(content={"status": "Random changes started"})


@app.post("/load_model")
async def load_model(data: dict):
    """
    Load a new model based on the provided log folder path.
    """
    global state_handler, current_model_folder
    current_model_folder = data["log_path"]
    config = ca_handler.load_model("train_log/" + data["log_path"])
    IMG_SIZE, COLOR_DIM, COND_DIM, x_tensor, current_input_image, cond = ca_handler.get_initial_data()
    state_handler = StateHandler(config, IMG_SIZE, COLOR_DIM, COND_DIM)
    state_handler.set_state(current_input_image[:, :, :3], x_tensor, cond)
    state_handler.set_seed_tensor(x_tensor.clone().detach())
    await reset_seed_and_condition()
    return JSONResponse(
        content={
            "status": "Model loaded successfully",
            "redirect_url": f"/{ca_handler.get_dataset_name()}",
        }
    )


@app.post("/adjust_image_size")
async def adjust_image_size(data: dict):
    """
    Adjust the image canvas and tensor size.
    """
    height, width = int(data["height"]), int(data["width"])
    print(f"Adjusting image size to {height}x{width}")
    with state_handler.get_lock():
        img_canvas = state_handler.get_img_canvas()
        img_tensor = state_handler.get_img_tensor()
        img_canvas = resize_canvas_centered(img_canvas, height, width)
        img_tensor = resize_tensor_centered(img_tensor, height, width)
        state_handler.set_img_tensor(img_tensor)
        state_handler.set_img_canvas(img_canvas)
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Image size adjusted"})


# --- MNIST Dataset Routes ---

@app.post("/paint")
async def paint(action: dict):
    """
    Paint on the image using a brush tool (for MNIST dataset).
    """
    with state_handler.get_lock():
        img_canvas = state_handler.get_img_canvas()
        img_tensor = state_handler.get_img_tensor().squeeze(0)
        img_canvas, img_tensor = paint_brush(
            img_canvas, img_tensor, action, img_canvas.shape[0]
        )
        state_handler.set_img_canvas(img_canvas)
        state_handler.set_img_tensor(img_tensor.unsqueeze(0))
    return JSONResponse(content={"status": "Painted"})


# --- Emoji Dataset Routes ---

@app.post("/roll_condition")
async def roll_condition():
    """
    Roll the condition for the emoji dataset.
    """
    _, _, _, _, _, cond = ca_handler.get_initial_data()
    with state_handler.get_lock():
        state_handler.set_condition_tensor(cond)
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Condition rolled"})


# --- Invertible Dataset Routes ---
@app.post("/invert_condition")
async def invert_condition():
    # return either [0, 1] or [1, 0] based on the current condition
    with state_handler.get_lock():
        tensor = state_handler.get_condition_tensor()
        if tensor.shape[1] == 2:  # Assuming condition tensor has shape [
            # Invert the condition tensor
            tensor = 1 - tensor
        else:
            # If the condition tensor is not binary, just return it as is
            tensor = tensor.clone()
        state_handler.set_condition_tensor(tensor)
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Condition inverted"})


@app.post("/change_condition")
async def change_condition(data: dict):
    """
    Change the condition for the emoji dataset.
    """
    print(f"Changing condition to {data}")
    condition_tensor = ca_handler.get_condition_tensor(int(data["index"]))
    with state_handler.get_lock():
        state_handler.set_condition_tensor(condition_tensor)
    await manager.broadcast_image(state_handler.get_img_canvas())
    return JSONResponse(content={"status": "Condition changed"})


# --- Dynamic Dataset Page ---
@app.get("/{dataset_type}")
async def dataset_page(request: Request, dataset_type: str):
    """
    Render the page for a specific dataset type.
    """

    train_folders = get_trainlog_folders()
    ui_config = ca_handler.get_ui_config()
    model_cfg = getattr(ca_handler.config, "MODEL", None)
    model_name = getattr(model_cfg, "NAME", "Unknown") if model_cfg else "Unknown"

    template_name = f"{dataset_type}.html"
    if not os.path.exists(os.path.join("app/templates", template_name)):
        template_name = "default.html"
    context = {
        "title": f"Dataset Page for {dataset_type}",
        "data": ui_config,
        "model_name": model_name,
        "model_overview": ui_config.get("MODEL_OVERVIEW"),
        "folders": train_folders,
        "current_model_folder": current_model_folder,
    }
    return templates.TemplateResponse(request, template_name, context)


@app.post("/set_speed")
async def set_speed(data: dict):
    """
    Set the speed (updates per second) for CA changes.
    """
    speed = float(data["speed"])
    with state_handler.get_lock():
        state_handler.set_speed(speed)
    return JSONResponse(content={"status": "Speed set"})
