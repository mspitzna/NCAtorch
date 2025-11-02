import numpy as np
from PIL import Image, ImageDraw

def delete_brush(img_canvas, img_tensor, action: dict, img_size: int):
    #print(f"shape of img_canvas: {img_canvas.shape}")
    #print(f"shape of img_tensor: {img_tensor.shape}")

    img_dim = img_canvas.shape[2]

    # Expecting JSON payload with x, y, brush_size, color
    x = int(action["x"])
    y = int(action["y"])

    # Rescale the x and y to the image size, action['x'] and action['y'] might be in range 0-512
    x = int(x * img_size / 512)
    y = int(y * img_size / 512)

    brush_size = int(action.get("brush_size", 10))
    color = (0, 0, 0) if img_dim == 3 else (0, 0, 0, 0)

    # Create a mask for the brush
    mask = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            x - brush_size // 2,
            y - brush_size // 2,
            x + brush_size // 2,
            y + brush_size // 2,
        ),
        fill=255,
    )
    mask = np.array(mask)

    # Apply the mask to img_canvas
    img_canvas[mask == 255] = color

    # rescale mask 256,256 to img_tensor shape 32, 32
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize((img_tensor.shape[2], img_tensor.shape[1]), resample=Image.NEAREST)
    mask = np.array(mask_img)
    print(f"rescaled mask shape: {mask.shape}, img_tensor shape: {img_tensor.shape}")

    print(f"mask unique values: {np.unique(mask)}, min mask value: {np.min(mask)}, max mask value: {np.max(mask)}")

    # Apply the mask to img_tensor
    for i in range(0, img_tensor.shape[0]):
        img_tensor[i][mask > 0] = 0

    return img_canvas, img_tensor


def paint_brush(img_canvas, img_tensor, action: dict, img_size: int):
    img_dim = img_canvas.shape[2]

    # Expecting JSON payload with x, y, brush_size, color
    x = int(action["x"])
    y = int(action["y"])

    # Rescale the x and y to the image size, action['x'] and action['y'] might be in range 0-512
    x = int(x * img_size / 512)
    y = int(y * img_size / 512)

    brush_size = int(action.get("brush_size", 10))
    color = tuple(action.get("color", (255, 255, 255)))

    # Create a mask for the brush
    mask = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            x - brush_size // 2,
            y - brush_size // 2,
            x + brush_size // 2,
            y + brush_size // 2,
        ),
        fill=255,
    )
    mask = np.array(mask)

    # Apply the mask to img_canvas
    #img_canvas[mask == 255] = color

    # Apply the mask to img_tensor
    for i in range(0, 1):
        img_tensor[i][mask == 255] = 1.0

    return img_canvas, img_tensor

def debug_delete_brush():
    import torch
    img_size = 168
    color_dim = 3
    # random img_canvas and img_tensor for debugging
    img_canvas = np.random.randint(0, 255, (img_size, img_size, color_dim), dtype=np.uint8)
    img_tensor = torch.rand((16, img_size, img_size))
    action = {
        "x": 256,
        "y": 256,
        "brush_size": 5,
        "color": [255, 255, 255]
    }

    img_canvas, img_tensor = delete_brush(img_canvas, img_tensor, action, img_size)

    # Save the img_canvas to check the result
    img_canvas_pil = Image.fromarray(img_canvas)
    img_canvas_pil.save("debug_img_canvas.png")

    # save each dim of the img_tensor as an image
    for i in range(img_tensor.shape[0]):
        img_tensor_pil = Image.fromarray((img_tensor[i].numpy() * 255).astype(np.uint8))
        img_tensor_pil.save(f"debug_img_tensor_{i}.png")


def debug_paint_brush():
    import torch
    img_size = 168
    color_dim = 3
    # random img_canvas and img_tensor for debugging
    img_canvas = np.random.randint(0, 255, (img_size, img_size, color_dim), dtype=np.uint8)
    img_tensor = torch.rand((16, img_size, img_size))
    action = {
        "x": 256,
        "y": 256,
        "brush_size": 5,
        "color": [255, 255, 255]
    }

    img_canvas, img_tensor = paint_brush(img_canvas, img_tensor, action, img_size)

    # Save the img_canvas to check the result
    img_canvas_pil = Image.fromarray(img_canvas)
    img_canvas_pil.save("debug_img_canvas.png")

    # save each dim of the img_tensor as an image
    for i in range(img_tensor.shape[0]):
        img_tensor_pil = Image.fromarray((img_tensor[i].numpy() * 255).astype(np.uint8))
        img_tensor_pil.save(f"debug_img_tensor_{i}.png")

if __name__ == "__main__":
    debug_paint_brush()