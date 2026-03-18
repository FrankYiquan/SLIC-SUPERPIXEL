import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt

def slic_custom(image, K=100, m=10, max_iter=10):
    """
    Simple Linear Iterative Clustering (SLIC) implementation from scratch.

    Parameters:
        image: input RGB image (H, W, 3)
        K: desired number of superpixels
        m: compactness parameter (higher = more square superpixels)
        max_iter: number of iterations

    Returns:
        labels: (H, W) array of superpixel assignments
    """

    # Convert to Lab color space
    img_lab = color.rgb2lab(image) # output: (H, W, 3) with L, a, b channels
    H, W = img_lab.shape[:2]

    # assume each k is a square that equally shared the image, compute the side length of square S
    N = H * W # area of the image
    S = int(np.sqrt(N / K)) # the side length

    # Initialize cluster centers on sqaure (non-overlap)
    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            L, a, b = img_lab[y, x]
            centers.append([L, a, b, x, y])

    centers = np.array(centers)
    num_centers = len(centers)

    # Step 4: Initialize label + distance maps
    labels = -np.ones((H, W), dtype=int)
    distances = np.full((H, W), np.inf)

    # k-means clustering
    # number of iterations determined by max_iter
    for _ in range(max_iter):
        # for each cluster center
        for i, (L_c, a_c, b_c, x_c, y_c) in enumerate(centers):

            # boundary of the square
            x_start = max(int(x_c - S), 0)
            x_end   = min(int(x_c + S), W)
            y_start = max(int(y_c - S), 0)
            y_end   = min(int(y_c + S), H)

            # within the local square, compute the distance to the cluster center and update labels and distances
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    L, a, b = img_lab[y, x]

                    # Color distance
                    dc = np.sqrt((L - L_c)**2 + (a - a_c)**2 + (b - b_c)**2)

                    # Spatial distance
                    ds = np.sqrt((x - x_c)**2 + (y - y_c)**2)

                    # Combined distance
                    D = np.sqrt(dc**2 + (ds / S)**2 * m**2)

                    # Update if this cluster center is closer
                    if D < distances[y, x]:
                        distances[y, x] = D
                        labels[y, x] = i

        # Update cluster centers
        new_centers = np.zeros_like(centers)
        counts = np.zeros(num_centers)

        # for each pixel, add its Lab and spatial values to the corresponding cluster center sum
        for y in range(H):
            for x in range(W):
                i = labels[y, x]
                L, a, b = img_lab[y, x]

                new_centers[i, :3] += [L, a, b]
                new_centers[i, 3:] += [x, y]
                counts[i] += 1

        # take the average to get the new cluster centers
        for i in range(num_centers):
            if counts[i] > 0:
                new_centers[i] /= counts[i]

        centers = new_centers
        distances.fill(np.inf)

    return labels