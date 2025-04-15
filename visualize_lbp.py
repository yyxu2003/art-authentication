#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced LBP Visualization for Van Gogh's Starry Night
- Author: yiyangxu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

image_path = "/Users/yiyangxu/Desktop/starry_night.jpg"

def visualize_enhanced_lbp(image_path, P=24, R=3, max_dim=2000):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    # Resize image to max_dim
    h, w = image.shape[:2]
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # Convert to RGB and Grayscale
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, P, R, method="uniform")

    # Normalize global LBP histogram
    hist_global, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    hist_global = hist_global.astype("float")
    hist_global /= (hist_global.sum() + 1e-7)

    # Compute regional histograms (4 quadrants)
    region_hists = []
    region_titles = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    half_h, half_w = new_h // 2, new_w // 2
    regions = [
        lbp[0:half_h, 0:half_w],
        lbp[0:half_h, half_w:],
        lbp[half_h:, 0:half_w],
        lbp[half_h:, half_w:]
    ]
    for region in regions:
        hist, _ = np.histogram(region.ravel(), bins=26, range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        region_hists.append(hist)

    # Compute LBP uniformity heatmap (std dev in 32x32 patches)
    patch_size = 32
    heatmap = np.zeros((new_h // patch_size, new_w // patch_size))
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            patch = lbp[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            heatmap[i, j] = np.std(patch)

    # Plot everything
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1.1])

    # Original
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(image_rgb)
    ax0.set_title("Original: The Starry Night", fontsize=14)
    ax0.axis('off')

    # LBP-coded image
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(lbp / lbp.max(), cmap='gray')
    ax1.set_title("LBP-Coded Image", fontsize=12)
    ax1.axis('off')

    # Global LBP histogram
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.bar(range(26), hist_global, color='#FFD700')
    ax2.set_title("Global LBP Histogram", fontsize=12)
    ax2.set_xlabel("Pattern Index")
    ax2.set_ylabel("Frequency")

    # Heatmap of LBP standard deviation (texture density)
    ax3 = fig.add_subplot(gs[1, 2])
    im = ax3.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax3.set_title("Textural Density Heatmap (Patchwise STD)", fontsize=12)
    ax3.axis('off')
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Regional histograms
    for i in range(4):
        ax = fig.add_subplot(gs[2 + i//2, i%2])
        ax.bar(range(26), region_hists[i], color='skyblue')
        ax.set_title(f"LBP Histogram – {region_titles[i]}", fontsize=12)
        ax.set_xlabel("Pattern Index")
        ax.set_ylabel("Frequency")

    fig.suptitle("LBP Analysis: Microtextures of Van Gogh's Starry Night", fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_enhanced_lbp(image_path)
