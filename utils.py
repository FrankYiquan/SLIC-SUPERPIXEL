import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.io import imsave
from skimage.util import img_as_float
import matplotlib.pyplot as plt


def compare_slic(image, custom_slic_fn, K=100, compactness=10):
    """
    Compare custom SLIC with built-in SLIC and save boundary images.

    Args:
        image: input image (H, W, 3)
        custom_slic_fn: SLIC function -> returns (H, W) label map
        K: number of superpixels
        compactness: SLIC compactness parameter
    """

    # Convert image to float (required by skimage)
    image_float = img_as_float(image)

    # Custom SLIC
    custom_labels = custom_slic_fn(image, K)

    # Built-in SLIC
    builtin_labels = slic(
        image_float,
        n_segments=K,
        compactness=compactness,
        start_label=0
    )

    # Draw boundaries
    custom_vis = mark_boundaries(image_float, custom_labels)
    builtin_vis = mark_boundaries(image_float, builtin_labels)

    # Save individual images
    imsave("images/output/custom_slic.png", (custom_vis * 255).astype(np.uint8))
    imsave("images/output/builtin_slic.png", (builtin_vis * 255).astype(np.uint8))

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(custom_vis)
    axes[1].set_title("Custom SLIC")
    axes[1].axis("off")

    axes[2].imshow(builtin_vis)
    axes[2].set_title("Built-in SLIC")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("images/output/comparison.png")
    plt.close()

    print("Saved results to images/output/")