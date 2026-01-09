"""
Test script to verify dataset structure and class mapping
"""
import os

# Check dataset structure
data_dir = 'data/Dataset/train'

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

if not os.path.exists(data_dir):
    print(f"❌ ERROR: Directory {data_dir} does not exist!")
else:
    print(f"✅ Directory found: {data_dir}\n")
    
    # Get folders in alphabetical order (same as TensorFlow)
    folders = sorted([d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
    
    print("Folders detected (alphabetical order - same as TensorFlow):")
    print("-" * 60)
    
    # Class mapping based on your requirements
    class_mapping = {
        'cloth': 'Cloth Mask',
        'n95': 'N95 Mask',
        'n95v': 'Partial Mask',
        'nfm': 'No Mask',
        'srg': 'Surgical Mask'
    }
    
    # Count images in each folder
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        num_images = len([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_label = class_mapping.get(folder, 'UNKNOWN')
        
        print(f"Index {idx}: '{folder}' → {class_label} ({num_images} images)")
    
    print("\n" + "=" * 60)
    print("CLASS NAMES LIST FOR CODE:")
    print("=" * 60)
    class_names_list = [class_mapping.get(folder, 'UNKNOWN') for folder in folders]
    print(f"self.class_names = {class_names_list}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    expected_folders = ['cloth', 'n95', 'n95v', 'nfm', 'srg']
    if folders == expected_folders:
        print("✅ All folders present and in correct order!")
    else:
        print("❌ Folder mismatch!")
        print(f"   Expected: {expected_folders}")
        print(f"   Found:    {folders}")
    
    # Verify total images
    total_images = sum([len([f for f in os.listdir(os.path.join(data_dir, folder))
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                       for folder in folders])
    print(f"✅ Total training images: {total_images}")
    
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("To train the model, run:")
    print("  python train_mask_model.py")
    print("\nOr with custom parameters:")
    print("  python train_mask_model.py --epochs 100 --batch_size 64")
    print("=" * 60)
