import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm

# Configuration
dataset_path = '    '
model_save_path = 'resnet_feature_extractor.pth'
kdtree_save_path = 'kdtree_index.pkl'
features_save_path = 'database_features.npz'

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load pre-trained ResNet18 and remove final classification layer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = models.resnet18(pretrained=True)
# Remove the final FC layer to get 512-dim feature vectors
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

print("Model loaded successfully")

# Load all images from dataset
image_files = sorted([f for f in os.listdir(dataset_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {len(image_files)} images in dataset")

# Extract features for all images
features_list = []
image_paths = []

print("Extracting features...")
with torch.no_grad():
    for img_file in tqdm(image_files):
        img_path = os.path.join(dataset_path, img_file)
        
        # Load and preprocess image
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Extract features
            features = model(img_tensor)
            features = features.squeeze().cpu().numpy()
            
            features_list.append(features)
            image_paths.append(img_file)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

# Convert to numpy array
features_array = np.array(features_list)
print(f"Extracted features shape: {features_array.shape}")

# Build KD-Tree for fast nearest neighbor search
print("Building KD-Tree...")
kdtree = KDTree(features_array, leaf_size=40, metric='euclidean')
print("KD-Tree built successfully")

# Save the model (feature extractor)
print(f"Saving model to {model_save_path}...")
torch.save(model.state_dict(), model_save_path)

# Save KD-Tree and associated data
print(f"Saving KD-Tree to {kdtree_save_path}...")
with open(kdtree_save_path, 'wb') as f:
    pickle.dump(kdtree, f)

# Save features and image paths for reference
print(f"Saving features to {features_save_path}...")
np.savez(features_save_path, 
         features=features_array,
         image_paths=np.array(image_paths))

print("\n=== Training Complete ===")
print(f"Model saved: {model_save_path}")
print(f"KD-Tree saved: {kdtree_save_path}")
print(f"Features saved: {features_save_path}")
print(f"Total images indexed: {len(image_paths)}")
