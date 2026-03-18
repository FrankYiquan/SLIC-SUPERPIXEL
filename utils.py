import numpy as np
from skimage.segmentation import slic, find_boundaries
from skimage.io import imsave
from skimage.util import img_as_float
from skimage.morphology import dilation, square
import matplotlib.pyplot as plt
import time


def compare_slic(image, custom_slic_fn, K=100, compactness=10):
    """
    Compare custom SLIC with built-in SLIC and save boundary images.
    """

    image_float = img_as_float(image)

    #custom SLIC
    start = time.time()
    custom_labels = custom_slic_fn(image, K=K, m=compactness)
    custom_time = time.time() - start

    # built-in SLIC
    start = time.time()
    builtin_labels = slic(
        image_float,
        n_segments=K,
        compactness=compactness,
        start_label=0
    )
    builtin_time = time.time() - start

    print(f"Custom SLIC Time:  {custom_time:.2f}s")
    print(f"Built-in SLIC Time: {builtin_time:.2f}s")

    # visualize boundaries
    def draw_boundaries(img, labels):
        boundaries = find_boundaries(labels, mode='outer')
        boundaries = dilation(boundaries, square(3))  # thickness control

        vis = img.copy()
        vis[boundaries] = [1, 0, 0]  # red
        vis = vis * 0.9              # darken background
        return vis

    custom_vis = draw_boundaries(image_float, custom_labels)
    builtin_vis = draw_boundaries(image_float, builtin_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(custom_vis)
    axes[0].set_title(f"Custom SLIC\n{custom_time:.2f}s")
    axes[0].axis("off")

    axes[1].imshow(builtin_vis)
    axes[1].set_title(f"Built-in SLIC\n{builtin_time:.2f}s")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"images/output/comparison_k{K}.png", dpi=300)
    plt.close()

    print("Saved results to images/output/")