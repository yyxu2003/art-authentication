#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 17:40:22 2025

@author: yiyangxu
"""

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Define dataset folder paths
dataset_path = "/Users/yiyangxu/vangogh_authenticator/data/raw/vgdataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Mapping folders to labels
label_map = {
    "0": 0,     # AI-generated and human forgeries
    "1": 1,     # Real Van Gogh painting
    "2": 2      # Other styles and paintings 
}

# List to store file paths and labels
image_paths = []
labels = []

# Loop through each folder and assign labels
for folder, label in label_map.items():
    folder_path = os.path.join(dataset_path, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure only images are included
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)

# Convert to DataFrame
df = pd.DataFrame({"image_path": image_paths, "label": labels})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())  # Preview the labeled dataset

# Split dataset (80% train, 20% test) while keeping class balance
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))

train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

print("Train and test sets saved successfully!")

# ---- ADDING IMAGE MOVEMENT BELOW ----

# Create train/test directories if they don’t exist
for folder in [train_path, test_path]:
    os.makedirs(folder, exist_ok=True)
    for category in label_map.keys():
        os.makedirs(os.path.join(folder, category), exist_ok=True)

# Move images into train/test folders
def move_images(df, target_folder):
    for _, row in df.iterrows():
        image_path = row["image_path"]
        category = os.path.basename(os.path.dirname(image_path))  # Get folder name (e.g., "0(ai)")
        new_path = os.path.join(target_folder, category, os.path.basename(image_path))
        shutil.move(image_path, new_path)  # Move the file

# Move training images
move_images(train_df, train_path)

# Move testing images
move_images(test_df, test_path)

print("images shuffled and moved to train/test folders successfully!")

def load_dataset(split='train'):
    """Load dataset from CSV"""
    df = pd.read_csv(f'{split}_set.csv')
    return df['image_path'].tolist(), df['label'].tolist()