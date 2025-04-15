import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# --------------------------
# Feature Extraction Core
# --------------------------

def extract_features_from_dataset(dataframe, output_csv, keep_image_path=False):
    """Process all images and save features with labels"""
    features = []
    labels = []
    image_paths = [] if keep_image_path else None

    for idx, row in dataframe.iterrows():
        try:
            img_path = row['image_path']
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"❌ Path not found: {img_path}")
            
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if img is None:
                raise ValueError(f"❌ Failed to load {img_path}")

            features.append(combined_features_with_brushstrokes(img))
            labels.append(row['label'])

            if keep_image_path:
                image_paths.append(img_path)

            if (idx + 1) % 50 == 0:
                print(f"✅ Processed {idx + 1}/{len(dataframe)} images")
                
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {str(e)}")

    # Create feature DataFrame
    feature_columns = [f"feat_{i}" for i in range(len(features[0]))]
    features_df = pd.DataFrame(features, columns=feature_columns)
    features_df["label"] = labels

    if keep_image_path:
        features_df.insert(0, "image_path", image_paths)

    features_df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv}")
    return features_df

# --------------------------
# Feature Components
# --------------------------

def van_gogh_hog_features(image, image_size=512):
    """Enhanced HOG features from LAB color space"""
    resized = cv2.resize(image, (image_size, image_size))
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # HOG parameters
    ppc = (16, 16)
    cpb = (2, 2)
    orientations = 9
    block_norm = 'L2-Hys'

    hog_l = hog(l, orientations=orientations, pixels_per_cell=ppc,
                cells_per_block=cpb, block_norm=block_norm)
    hog_a = hog(a, orientations=orientations, pixels_per_cell=ppc,
                cells_per_block=cpb, block_norm=block_norm)
    hog_b = hog(b, orientations=orientations, pixels_per_cell=ppc,
                cells_per_block=cpb, block_norm=block_norm)

    return 0.7*hog_l + 0.15*hog_a + 0.15*hog_b

def compute_color_histograms(img, bins=16):
    """Multi-color space histograms (RGB + HSV)"""
    hist_features = []
    
    # RGB Histograms
    rgb_hist = []
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        rgb_hist.append(hist)
    hist_features.extend(rgb_hist)
    
    # HSV Histograms
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 180] if i == 0 else [0, 256])
        hist = cv2.normalize(hist, None).flatten()
        hist_features.append(hist)
    
    return np.concatenate(hist_features)

def compute_brushstroke_features(image):
    """Quantify painting stroke characteristics"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Adaptive edge detection
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Stroke statistics
    _, _, stats, _ = cv2.connectedComponentsWithStats(edges)
    stroke_lengths = stats[1:, 4]  # Ignore background
    
    return [
        np.mean(edges)/255,          # Edge density
        np.std(stroke_lengths),      # Stroke variation
        np.percentile(stroke_lengths, 90)  # Long stroke ratio
    ]

# --------------------------
# Feature Integration
# --------------------------

def combined_features_with_brushstrokes(image):
    """Integrated feature pipeline"""
    resized = cv2.resize(image, (512, 512))
    
    # Core features
    hog_feats = van_gogh_hog_features(resized)
    color_feats = compute_color_histograms(resized)
    lbp_feats = compute_lbp_features(resized)
    # line_feats = compute_line_features(resized)
    # brush_feats = compute_brushstroke_features(resized)
    
    # Combine all features (brush_feats temporarily excluded)
    return np.concatenate([
        hog_feats,
        color_feats,
        lbp_feats,
        # line_feats,
        # brush_feats
    ])


# --------------------------
# Support Functions
# --------------------------

def compute_lbp_features(image, P=24, R=3):
    """Texture analysis via Local Binary Patterns"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist = np.histogram(lbp.ravel(), bins=26)[0]
    return hist / hist.sum()

# def compute_line_features(image):
#     """Detect and count directional lines"""
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
#                            threshold=80, minLineLength=30, maxLineGap=5) 
    
#     counts = [0, 0, 0, 0]  # horizontal, vertical, 45°, -45°
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
#             if -15 < angle < 15 or 165 < abs(angle) < 180:
#                 counts[0] += 1
#             elif 75 < angle < 105 or -105 < angle < -75:
#                 counts[1] += 1
#             elif 30 < angle < 60:
#                 counts[2] += 1
#             elif -60 < angle < -30:
#                 counts[3] += 1
                
#     return np.array(counts, dtype=np.float32)

# def compute_stroke_features(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
    
#     # Stroke direction variance
#     gradients_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0)
#     gradients_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1)
#     direction_variance = np.var(np.arctan2(gradients_y, gradients_x))
    
#     # Stroke length distribution
#     _, _, stats, _ = cv2.connectedComponentsWithStats(edges)
#     return [direction_variance, np.percentile(stats[1:, 4], 90)]

# --------------------------
# Validation & Pipeline
# --------------------------

def validate_feature_extraction(image_path):
    """Full feature validation workflow"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError("Image not loaded")

    resized = cv2.resize(image, (512, 512))
    features = combined_features_with_brushstrokes(resized)
    
    print("\n=== Feature Validation Report ===")
    print(f"Total dimensions: {features.shape[0]}")
    print("Feature ranges:")
    print(f"- Min: {features.min():.4f}")
    print(f"- Max: {features.max():.4f}")
    print(f"- Mean: {features.mean():.4f}")
    print(f"- NaN values: {np.isnan(features).sum()}")
    
    expected_dim = 34596 + 96 + 26 + 4  # Brushstroke features excluded (previously +3)
  # Updated with brushstroke
    assert features.shape[0] == expected_dim, \
        f"Dimension mismatch: Expected {expected_dim}, got {features.shape[0]}"
    print("\n✅ All features validated successfully")

# --------------------------
# ML Pipeline
# --------------------------

pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=300),
    SVC(kernel='rbf', class_weight='balanced', probability=True)
)

# --------------------------
# Run Feature Extraction
# --------------------------

train_df = pd.read_csv('/Users/yiyangxu/vangogh_authenticator/data/processed/train_set.csv')
train_features_df = extract_features_from_dataset(train_df, 'train_features.csv')

# test_df = pd.read_csv('/Users/yiyangxu/vangogh_authenticator/data/processed/test_set.csv')
# test_features_df = extract_features_from_dataset(test_df, 'test_features.csv')





