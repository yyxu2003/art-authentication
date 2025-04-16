#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:21:39 2025

@author: yiyangxu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from matplotlib.patches import Rectangle, Patch

def compute_and_visualize_hog(image_path):
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image from: {image_path}")
        return

    # Preserve aspect ratio
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    image_resized = cv2.resize(image, (int(w*scale), int(h*scale)))
    
    # Color conversions
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    # HOG histogram for selected region
    x0, y0 = int(0.3 * w * scale), int(0.4 * h * scale)
    x1, y1 = int(0.2 * w * scale), int(0.15 * h * scale)
    patch = l[y0:y0 + y1, x0:x0 + x1]

    hog_features, _ = hog(patch, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(1, 1), visualize=True, feature_vector=True)

    # Convert flattened vector to 9-bin histogram
    hog_hist = np.zeros(9)
    for i in range(0, len(hog_features), 9):
        hog_hist += hog_features[i:i+9]

    hog_hist /= hog_hist.sum()  # normalize

    # Enhanced HOG color map
    def get_hog_vis(channel):
        _, hog_image = hog(channel, orientations=9, pixels_per_cell=(32, 32),
                          cells_per_block=(2, 2), block_norm='L2-Hys',
                          visualize=True, feature_vector=True)
        hog_norm = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
        colored_hog = np.zeros((*hog_image.shape, 3))
        bin_size = 1/9
        for ori in range(9):
            mask = (hog_norm >= ori*bin_size) & (hog_norm < (ori+1)*bin_size)
            colored_hog[mask] = plt.cm.hsv(ori/9)[:3]
        return colored_hog

    # Create figure
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 0.5, 0.6])

    # Original image
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(image_rgb)
    ax0.add_patch(Rectangle(
        (x0, y0), x1, y1,
        linewidth=2, edgecolor='yellow', facecolor='none'
    ))
    ax0.set_title("Van Gogh's Brushstroke Analysis: The Starry Night", fontsize=16)
    ax0.axis('off')

    # LAB channels
    lab_titles = ['Lightness (Texture)', 'Green-Red (Color)', 'Blue-Yellow (Color)']
    for i, (channel, title) in enumerate(zip([l, a, b], lab_titles)):
        ax = fig.add_subplot(gs[1, i])
        cmap = 'gray' if i==0 else 'coolwarm' if i==1 else 'RdYlBu'
        ax.imshow(channel, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # HOG features
    hog_imgs = [get_hog_vis(l), get_hog_vis(a), get_hog_vis(b)]
    for i, hog_img in enumerate(hog_imgs):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(hog_img)
        ax.set_title(f"HOG: {lab_titles[i].split(' ')[0]}", fontsize=12)
        ax.axis('off')

    # Orientation legend
    legend_elements = [Patch(facecolor=plt.cm.hsv(i/9), 
                      label=f"{i*20}°") for i in range(9)]
    fig.legend(handles=legend_elements, ncol=9,
              title="Brushstroke Orientation Directions",
              loc='lower center', fontsize=10)

    # HOG histogram bar plot
    ax_hist = fig.add_subplot(gs[3, 1])
    bin_labels = [f"{i*20}°" for i in range(9)]
    ax_hist.bar(bin_labels, hog_hist, color=[plt.cm.hsv(i/9) for i in range(9)])
    ax_hist.set_title("HOG Orientation Histogram for Yellow Box Region", fontsize=12)
    ax_hist.set_ylabel("Normalized Magnitude")
    ax_hist.set_ylim(0, hog_hist.max()*1.2)

    plt.tight_layout()
    plt.show()

compute_and_visualize_hog("/Users/yiyangxu/Desktop/starry_night.jpg")
