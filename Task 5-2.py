import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Load saved components
model_save_path = 'resnet_feature_extractor.pth'
kdtree_save_path = 'kdtree_index.pkl'
features_save_path = 'database_features.npz'

# Setup device and transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet18(pretrained=False)
model = nn.Sequential(*list(model.children())[:-1])
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)
model.eval()

# Load KD-Tree
with open(kdtree_save_path, 'rb') as f:
    kdtree = pickle.load(f)

# Load database features and paths
data = np.load(features_save_path)
database_features = data['features']
image_paths = data['image_paths']

print(f"Loaded {len(image_paths)} images from database")

def find_similar_images(query_image_path, k=5):
    """Find k most similar images to the query image."""
    
    # Load and preprocess query image
    img = Image.open(query_image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        query_features = model(img_tensor)
        query_features = query_features.squeeze().cpu().numpy()
    
    # Query KD-Tree for nearest neighbors
    distances, indices = kdtree.query([query_features], k=k)
    
    # Return results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'rank': i + 1,
            'image_path': './vpr_dataset/smaller_database/' + image_paths[idx],
            'distance': dist
        })
    
    return results
    
def display_results(query_path, results):
    """Display the query image and top k similar images."""
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 4))
    
    # Display query image
    query_img = Image.open(query_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Display retrieved images
    for i, result in enumerate(results):
        img = Image.open(result['image_path']).convert('RGB')
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Rank {result['rank']}\nDist: {result['distance']:.4f}", 
                              fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    query_path = './vpr_dataset/query/query_1_1.jpg'
    
    print(f"Finding similar images to: {query_path}")
    results = find_similar_images(query_path, k=5)
    
    print("\nTop 5 most similar images:")
    for result in results:
        print(f"{result['rank']}. {result['image_path']} (distance: {result['distance']:.4f})")

    display_results(query_path, results)
