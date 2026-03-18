import numpy as np
from skimage import color
from skimage.segmentation import relabel_sequential
import numpy as np

def compute_gradient(img_lab):
    grad = np.zeros(img_lab.shape[:2])
    grad[1:-1, 1:-1] = (
        np.sum((img_lab[1:-1, 2:] - img_lab[1:-1, :-2])**2, axis=2) +
        np.sum((img_lab[2:, 1:-1] - img_lab[:-2, 1:-1])**2, axis=2)
    )
    return grad

def slic_custom(image, K=4000, m=30, max_iter=10):
    """
    Custom SLIC implementation:
    - Proper SLIC distance
    - Stable iteration
    - Tuned for large images
    """

    # Convert to float [0,1]
    if image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0

    # Convert to Lab
    img_lab = color.rgb2lab(image)
    H, W = img_lab.shape[:2]

    # Compute grid interval
    N = H * W
    S = int(np.sqrt(N / K))

    # Gradient for center adjustment
    grad = compute_gradient(img_lab)

    # Initialize centers
    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):

            y0, x0 = y, x
            min_grad = grad[y, x]

            # Move to lowest gradient in 3x3
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if grad[ny, nx] < min_grad:
                            min_grad = grad[ny, nx]
                            y0, x0 = ny, nx

            L, a, b = img_lab[y0, x0]
            centers.append([L, a, b, x0, y0])

    centers = np.array(centers, dtype=np.float64)
    num_centers = len(centers)

    # Initialize maps
    labels = -np.ones((H, W), dtype=np.int32)
    distances = np.full((H, W), np.inf)

    ys_full, xs_full = np.indices((H, W))

    # iteration 
    for _ in range(max_iter):

       
        labels.fill(-1)
        distances.fill(np.inf)

        for i, (L_c, a_c, b_c, x_c, y_c) in enumerate(centers):

            x_start = max(int(x_c - S), 0)
            x_end   = min(int(x_c + S), W)
            y_start = max(int(y_c - S), 0)
            y_end   = min(int(y_c + S), H)

            region = img_lab[y_start:y_end, x_start:x_end]
            xs = xs_full[y_start:y_end, x_start:x_end]
            ys = ys_full[y_start:y_end, x_start:x_end]

            # Color distance
            dc2 = (
                (region[..., 0] - L_c) ** 2 +
                (region[..., 1] - a_c) ** 2 +
                (region[..., 2] - b_c) ** 2
            )

            # Spatial distance
            ds2 = (xs - x_c) ** 2 + (ys - y_c) ** 2

            # SLIC distance
            D2 = dc2 + (m ** 2 / S ** 2) * ds2

            sub_dist = distances[y_start:y_end, x_start:x_end]
            sub_labels = labels[y_start:y_end, x_start:x_end]

            mask = D2 < sub_dist
            sub_dist[mask] = D2[mask]
            sub_labels[mask] = i

        # --- Update centers ---
        flat_labels = labels.ravel()
        valid = flat_labels >= 0
        flat_labels = flat_labels[valid]

        flat_L = img_lab[..., 0].ravel()[valid]
        flat_a = img_lab[..., 1].ravel()[valid]
        flat_b = img_lab[..., 2].ravel()[valid]
        flat_x = xs_full.ravel()[valid]
        flat_y = ys_full.ravel()[valid]

        counts = np.bincount(flat_labels, minlength=num_centers)

        sum_L = np.bincount(flat_labels, weights=flat_L, minlength=num_centers)
        sum_a = np.bincount(flat_labels, weights=flat_a, minlength=num_centers)
        sum_b = np.bincount(flat_labels, weights=flat_b, minlength=num_centers)
        sum_x = np.bincount(flat_labels, weights=flat_x, minlength=num_centers)
        sum_y = np.bincount(flat_labels, weights=flat_y, minlength=num_centers)

        nonzero = counts > 0

        centers[nonzero, 0] = sum_L[nonzero] / counts[nonzero]
        centers[nonzero, 1] = sum_a[nonzero] / counts[nonzero]
        centers[nonzero, 2] = sum_b[nonzero] / counts[nonzero]
        centers[nonzero, 3] = sum_x[nonzero] / counts[nonzero]
        centers[nonzero, 4] = sum_y[nonzero] / counts[nonzero]

    # Relabel for clean output
    labels, _, _ = relabel_sequential(labels)

    return labels