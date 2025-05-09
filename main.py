# %%
from collections import defaultdict
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import torch
import tqdm
import skimage.restoration
import nerfpaint

# %%

device = nerfpaint.set_default_tensor_type()


def tv_preprocess(image, mask):
    """Normalize and convert image and mask to PyTorch tensors."""
    image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    img_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    return img_tensor, mask_tensor


def tv_loss(x):
    """Compute anisotropic total variation loss."""
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean()


def inpaint_tv(image, mask, num_iters=200, lr=0.1, lambda_tv=0.1):
    """
    Inpaint using TV minimization via gradient descent.
    """
    losses = []
    img_tensor, mask_tensor = tv_preprocess(image, mask)
    inpaint = img_tensor.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([inpaint], lr=lr)

    for _ in tqdm.trange(num_iters, desc="TV Inpainting"):
        optimizer.zero_grad()

        data_term = ((1 - mask_tensor) * (inpaint - img_tensor)).pow(2).mean()
        tv_term = tv_loss(inpaint)

        loss = data_term + lambda_tv * tv_term
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            inpaint.clamp_(0.0, 1.0)

    result = inpaint.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    result = (result * 255).astype(np.uint8)

    if result.shape[2] == 1:
        result = result[:, :, 0]

    # plot the loss curve
    plt.rcParams.update({"font.size": 10})
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curve For TV Inpainting")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return result


def inpaint_ns(image, mask):
    return cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_NS)


def inpaint_fmm(image, mask):
    return cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)


def inpaint_biharmonic(image, mask):
    channel_axis = -1 if image.ndim == 3 else None
    inpainted_f64 = (
        skimage.restoration.inpaint_biharmonic(image, mask, channel_axis=channel_axis)
        * 255.0
    )
    return inpainted_f64.clip(0, 255).astype(np.uint8)


def display_inpainting_method(image, mask, method_title, method_func):
    # create a masked image
    masked_image = image.copy()
    masked_image[mask == 1] = 0

    # in-paint the image
    inpainted = method_func(masked_image, mask)
    inpainted[mask == 0] = image[mask == 0]

    # get the PSNR and SSIM scores
    psnr_score = cv.PSNR(image, inpainted)
    ssim_score = ssim(image, inpainted, channel_axis=-1)

    # Create a single row plot with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(left=0.1, right=0.9)

    # set font size
    plt.rcParams.update({"font.size": 14})

    axes[0].set_title("Ground Truth Image")
    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")

    axes[1].set_title("Masked Image")
    axes[1].imshow(masked_image, cmap="gray")
    axes[1].axis("off")

    axes[2].set_title(
        f"{method_title} (PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.2f})"
    )
    axes[2].imshow(inpainted, cmap="gray")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # Compute metrics
    abs_diff = np.abs(image.astype(np.float32) - inpainted.astype(np.float32))
    _, ssim_map = ssim(image, inpainted, channel_axis=-1, full=True)

    # Combined plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image.astype(np.float32), cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(
        abs_diff,
    )
    plt.title("Absolute Difference")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(
        ssim_map,
    )
    plt.title("SSIM Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return psnr_score, ssim_score, inpainted


mono_to_rgb = lambda x: np.repeat(x[:, :, np.newaxis], 3, axis=2)
rgb_to_mono = lambda x: np.mean(x, axis=2)


def inpaint_nerf(image, mask, l=4, epochs=1500):
    image = image.astype(np.float32) / 255.0

    is_rgb = image.ndim == 3 and image.shape[2] == 3
    assert is_rgb or image.ndim == 2, "Image must be RGB or grayscale."

    if not is_rgb:
        image = mono_to_rgb(image)

    y, losses = nerfpaint.inpaint_nerf(
        image=image,
        mask=mask.astype(np.uint8),
        device=device,
        epochs=epochs,
        l=l,
    )

    if not is_rgb:
        y = rgb_to_mono(y)

    y = (y * 255).clip(0, 255).astype(np.uint8)

    # plot the loss curve
    plt.rcParams.update({"font.size": 10})
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss Curve For 2D NeRF Inpainting")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return y


def get_mask(h: int, w: int):
    mask = np.zeros((h, w), dtype=np.uint8)
    h_1o16 = h // 16
    w_1o16 = w // 16
    h_1o64 = h // 64
    w_1o64 = w // 64
    boxes = [
        [h_1o16 * 3, h_1o16 * 4, w_1o16, w - w_1o16 * 2],
        [h_1o16 * 1, h_1o16 * 15, w_1o16 * 8, w_1o16 * 9],
        [h_1o16 * 7, h_1o16 * 8, w_1o16 * 0, w_1o16 * 14],
        [h_1o16 * 0, h_1o16 * 16, w_1o16 * 12, w_1o16 * 13],
        [h_1o64 * 40, h_1o64 * 41, w_1o16 * 0, w_1o16 * 14],
        [h_1o16 * 0, h_1o16 * 16, w_1o64 * 40, w_1o64 * 41],
        [h_1o16 * 0, h_1o16 * 16, w_1o64 * 20, w_1o64 * 21],
    ]
    for y0, y1, x0, x1 in boxes:
        mask[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] + 1) % 2

    return mask


if __name__ == "__main__":
    h, w = 512, 512
    mask = get_mask(h, w)
    test_image_paths = [
        "boat.png",
        "tulips.png",
        "baboon.png",
        "peppers.png",
        "HappyFish.jpg",
    ]
    test_image_paths = [
        (i + 1, Path("test_images") / p) for i, p in enumerate(test_image_paths)
    ]

    all_results = {}
    for nerf_l in range(1, 5):
        results_dict = defaultdict(list)
        methods = {
            f"NeRF (l={nerf_l})": lambda x, y: inpaint_nerf(
                x, y, l=nerf_l, epochs=1500
            ),
            "Biharmonic": inpaint_biharmonic,
            "FMM": inpaint_fmm,
            "NS": inpaint_ns,
            "TV": inpaint_tv,
        }
        results_dict["Method"].extend(methods.keys())
        for image_id, image_path in test_image_paths:
            image = np.array(Image.open(image_path), dtype=np.uint8)
            image = cv.resize(image, (h, w), interpolation=cv.INTER_LINEAR)

            psnrs = []
            ssims = []
            for method_name, method_func in methods.items():
                val_psnr, val_ssim, img = display_inpainting_method(
                    image, mask, method_name, method_func
                )
                psnrs.append(val_psnr)
                ssims.append(val_ssim)

            results_dict[f"PSNRs ({image_id})"].extend(psnrs)
            results_dict[f"SSIMs ({image_id})"].extend(ssims)

        df_results = pd.DataFrame(results_dict)
        df_results.to_csv(f"results-{nerf_l}.csv", index=False)
        all_results[nerf_l] = df_results

# %%
import matplotlib.animation as animation

# Load data
nerf_res = np.load("data/nerf-res.npz")
psnrs = nerf_res["psnrs"]
ssims = nerf_res["ssims"]
imgs = nerf_res["imgs"]

# Create figure for animation
fig, ax = plt.subplots()
im = ax.imshow(imgs[0], cmap="gray")
title = ax.set_title("")
ax.axis("off")


def update(frame):
    im.set_data(imgs[frame])
    title.set_text(
        f"NeRF $L$={frame+1} (PSNR: {psnrs[frame]:.2f}, SSIM: {ssims[frame]:.2f})"
    )
    return [im, title]


ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=300, blit=True)

# Save as GIF (requires ImageMagick or Pillow installed)
ani.save("nerf_animation.gif", writer="pillow")
plt.close()
# %%
# Define x-axis as levels of L
# Define x-axis values from 0 to 14
L_vals = np.arange(0, 15)

# Ensure psnrs and ssims are properly padded or sliced to match L_vals
# If psnrs and ssims have fewer than 15 entries, pad with NaNs for plotting
psnrs_plot = np.full_like(L_vals, np.nan, dtype=np.float64)
ssims_plot = np.full_like(L_vals, np.nan, dtype=np.float64)
psnrs_plot[: len(psnrs)] = psnrs
ssims_plot[: len(ssims)] = ssims

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# PSNR plot
axs[0].plot(L_vals, psnrs_plot, marker="x")
axs[0].set_title("PSNR vs. L")
axs[0].set_xlabel("L")
axs[0].set_ylabel("PSNR (dB)")
axs[0].set_xticks(L_vals)
axs[0].grid(True)

# SSIM plot
axs[1].plot(L_vals, ssims_plot, marker="x")
axs[1].set_title("SSIM vs. L")
axs[1].set_xlabel("L")
axs[1].set_ylabel("SSIM")
axs[1].set_xticks(L_vals)
axs[1].grid(True)

plt.tight_layout()
plt.show()
