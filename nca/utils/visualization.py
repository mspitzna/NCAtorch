import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from nca.utils.image_utils import to_rgb
from PIL import Image
import matplotlib.transforms as transforms

def visualize_batch(
    x0, 
    x, 
    target, 
    x_state_logs,           # dict: {step_id: tensor}
    step_i,
    plot=False,
    save_path=".",
    max_image_dim=4048,
    pixel_offset=-80,        # negative => left of the axes
    grayscale=False
):
    """
    Shows how to place row labels with a fixed pixel offset 
    to the left of the axes, regardless of figure size.
    """

    ######### 1) Gather row data and titles #########
    def prep(tensor):
        if grayscale:
            return to_rgb(tensor[:, 0:1].detach().cpu()) # TODO
        return to_rgb(tensor[:, :target.shape[1]].detach().cpu())

    row_titles = ["Seed"]
    sorted_keys = sorted(x_state_logs.keys())
    row_titles += [f"Step {k}" for k in sorted_keys]
    row_titles += ["Pred.", "Target"]
    row_titles = [title for title in reversed(row_titles)]

    frames = [prep(x0)]
    for k in sorted_keys:
        frames.append(prep(x_state_logs[k]))
    frames.append(prep(x))
    frames.append(prep(target))

    ######### 2) Make the grid (N rows, each row has batch_size images) #########
    vis = torch.cat(frames, dim=0)  # shape [N * B, C, H, W]
    batch_size = x0.shape[0]
    grid = vutils.make_grid(vis, nrow=batch_size, padding=4, normalize=False, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).numpy()  # [H, W, C]
    grid_np = np.clip(grid_np, 0, 1)

    ######### 3) Downscale if huge #########
    h, w, _ = grid_np.shape
    scale = min(max_image_dim / float(h), max_image_dim / float(w))
    if scale < 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_pil = Image.fromarray((grid_np * 255).astype(np.uint8))
        img_pil = img_pil.resize((new_w, new_h), resample=Image.BILINEAR)
        grid_np = np.array(img_pil).astype(np.float32) / 255.0
        h, w, _ = grid_np.shape

    ######### 4) Plot the image #########
    fig_height = 6
    fig_width = fig_height * (w / float(h))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(grid_np)
    ax.axis("off")

    ######### 5) Prepare a transform that adds a pixel offset to 'axes fraction' #########
    #  - 'ax.transAxes'   => x,y in [0..1] across the entire axes
    #  - 'fig.dpi_scale_trans' converts *pixel* offsets into figure-relative.
    #  - 'ScaledTranslation(pixel_offset / fig.dpi, 0, fig.dpi_scale_trans)'
    #      means "move left by 'pixel_offset' in screen pixels."
    offset_transform = transforms.ScaledTranslation(
        pixel_offset / fig.dpi,   # convert pixels to "figure inches"
        0,
        fig.dpi_scale_trans
    )
    # Combine them:
    text_transform = ax.transAxes + offset_transform

    ######### 6) Place row labels #########
    # Each row is 1 / row_count in axes fraction. 
    row_count = len(row_titles)
    # Usually, row=0 is at the *top* if your image is in normal orientation;
    # might need to invert if the final output is flipped.
    for i, label in enumerate(row_titles):
        # fraction from top to bottom
        y_frac = (i + 0.5) / row_count  
        # If your image is inverted, do 1 - ...
        # y_frac = 1.0 - (i + 0.5)/row_count

        ax.text(
            0,               # x=0 at left edge of the axes
            y_frac, 
            label,
            color="black",
            fontsize=12,
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            transform=text_transform,  # Use axes-fraction + pixel offset
            clip_on=False
        )

    ######### 7) Show or save #########
    if plot:
        plt.show()

    out_path = os.path.join(save_path, f"step_{step_i}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return grid_np

def save_image(image, save_path):
    """
    Save an image using tochvision.utils.save_image
    """
    vutils.save_image(to_rgb(image), save_path)