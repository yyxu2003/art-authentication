#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:40:56 2025

@author: yiyangxu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/Users/yiyangxu/Desktop/starry_night.jpg"

def compute_and_plot_color_histograms(image_path, bins=16):
    # Load and convert image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)

    # Prepare figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Color Histogram Visualization (RGB + HSV)", fontsize=16)

    channels_rgb = ['Red', 'Green', 'Blue']
    channels_hsv = ['Hue', 'Saturation', 'Value']
    colors = ['r', 'g', 'b']

    rgb_histograms = []
    hsv_histograms = []

    # Plot RGB histograms
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_resized], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        rgb_histograms.append(hist)
        axs[0, i].bar(range(bins), hist, color=color)
        axs[0, i].set_title(f"RGB - {channels_rgb[i]}")
        axs[0, i].set_xlim([0, bins])

    # Plot HSV histograms
    for i in range(3):
        channel_range = [0, 180] if i == 0 else [0, 256]
        hist = cv2.calcHist([image_hsv], [i], None, [bins], channel_range)
        hist = cv2.normalize(hist, None).flatten()
        hsv_histograms.append(hist)
        axs[1, i].bar(range(bins), hist, color='gray')
        axs[1, i].set_title(f"HSV - {channels_hsv[i]}")
        axs[1, i].set_xlim([0, bins])

    plt.tight_layout()
    plt.show()

    # ➕ Concatenate all histograms into a single feature vector
    color_feature_vector = np.concatenate(rgb_histograms + hsv_histograms)

    # 🖨 Show a preview of the vector
    print("\n🎯 Sample of the 96-dimensional color feature vector (first 20 values):")
    print(color_feature_vector[:20])

    print(f"\n✅ Full feature vector shape: {color_feature_vector.shape}")


compute_and_plot_color_histograms(image_path)
