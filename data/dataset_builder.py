#!/usr/bin/env python3
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_path = "/Users/yiyangxu/vangogh_authenticator/data/raw/vgdataset"
processed_path = "/Users/yiyangxu/vangogh_authenticator/data/processed"

train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Ensure processed directory exists
os.makedirs(processed_path, exist_ok=True)

# def augment_vangogh(class_dir, target_count):
#     """Augment Van Gogh images to reach target count in their original directory"""
#     datagen = ImageDataGenerator(
#         rotation_range=15,
#         width_shift_range=0.1,
#         brightness_range=[0.9, 1.1],
#         horizontal_flip=True
#     )

#     existing = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#     needed = target_count - existing

#     if needed > 0:
#         print(f"\n🖌️ Augmenting {class_dir}:")
#         print(f"Existing: {existing} | Needed: {needed} | Target: {target_count}")
        
#         generator = datagen.flow_from_directory(
#             os.path.dirname(class_dir),
#             classes=[os.path.basename(class_dir)],
#             target_size=(512, 512),
#             batch_size=32,
#             save_to_dir=class_dir,
#             save_prefix='aug',
#             save_format='jpg',
#             shuffle=False
#         )
        
#         # Generate exactly the needed number of images
#         batches_needed = (needed + generator.batch_size - 1) // generator.batch_size
#         for _ in range(batches_needed):
#             next(generator) 

def create_simple_csv(split_dir, csv_name):
    """Create CSV listing all images in a directory split"""
    image_paths = []
    labels = []
    
    for class_label in ["0", "1"]:
        class_path = os.path.join(split_dir, class_label)
        for img in os.listdir(class_path):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_path, img))
                labels.append(int(class_label))
    
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    csv_path = os.path.join(processed_path, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"✅ Created {csv_path} with {len(df)} entries")

def verify_counts():
    """Verify final image counts in each directory"""
    print("\n🔍 Final Counts:")
    for split in ["train", "test"]:
        for cls in ["0", "1"]:
            path = os.path.join(dataset_path, split, cls)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"{split}/class {cls}: {count} images")

if __name__ == "__main__":
    # # Double Van Gogh images in both train and test
    # augment_vangogh(os.path.join(train_dir, "1"), target_count=200)  # 100 -> 200
    # augment_vangogh(os.path.join(test_dir, "1"), target_count=50)    # 25 -> 50
    
    # Create CSVs without any balancing
    create_simple_csv(train_dir, "train_set.csv")
    create_simple_csv(test_dir, "test_set.csv")
    
    # Verification
    verify_counts()
