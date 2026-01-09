# Face Mask Dataset Information

## Dataset Structure

The dataset is now located at: `data/Dataset/`

```
data/Dataset/
├── train/
│   ├── cloth/      (396 images) → Cloth Mask
│   ├── n95/        (354 images) → N95 Mask
│   ├── n95v/       (390 images) → Partial Mask (N95 with valve)
│   ├── nfm/        (474 images) → No Mask
│   └── srg/        (342 images) → Surgical Mask
└── test/
    ├── cloth/      → Cloth Mask
    ├── n95/        → N95 Mask
    ├── n95v/       → Partial Mask
    ├── nfm/        → No Mask
    └── srg/        → Surgical Mask
```

**Total Training Images**: 1,956  
**Total Dataset Images**: 2,286

## Class Mapping

| Folder Name | Class Label      | Description                    |
|-------------|------------------|--------------------------------|
| `cloth`     | Cloth Mask       | Cloth face mask                |
| `n95`       | N95 Mask         | N95 respirator mask            |
| `n95v`      | Partial Mask     | N95 mask with valve            |
| `nfm`       | No Mask          | No face mask worn              |
| `srg`       | Surgical Mask    | Surgical/medical face mask     |

## Training the Model

The `train_mask_model.py` script has been updated to work with this dataset structure.

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Train with default settings (50 epochs)
python train_mask_model.py

# Train with custom settings
python train_mask_model.py --epochs 100 --batch_size 64

# Train with custom data directory
python train_mask_model.py --data_dir data/Dataset/train --epochs 75
```

### Command Line Arguments

- `--data_dir`: Path to training data directory (default: `data/Dataset/train`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--validation_split`: Validation split ratio (default: 0.2)
- `--save_path`: Path to save the trained model (default: `models/mask_detection_model.h5`)

## Model Details

- **Input Shape**: 128x128x3 (RGB images)
- **Number of Classes**: 5
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Output**: Softmax classification over 5 classes

## Data Augmentation

The training script applies the following augmentation techniques:
- Rotation (±30 degrees)
- Width/Height shift (±30%)
- Shear transformation (20%)
- Zoom (±30%)
- Horizontal flip
- Brightness variation (80-120%)
- Channel shift (±20)

## Output

After training, you will get:
- Trained model saved as `.h5` file in `models/` directory
- Training history plot saved as `training_history.png`
- TensorBoard logs in `logs/` directory
- Best model checkpoint based on validation accuracy

## Using the Trained Model

Once trained, the model can be loaded and used for inference:

```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the model architecture (same as in train_mask_model.py)
class MaskDetectionCNN(nn.Module):
    # ... (copy architecture from train_mask_model.py)
    pass

# Load the trained model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MaskDetectionCNN(num_classes=5)
checkpoint = torch.load('models/mask_detection_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
img = Image.open('path/to/image.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, class_idx = torch.max(probabilities, 1)

class_names = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
predicted_class = class_names[class_idx.item()]

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence.item():.2%}")
```

## Source

Dataset downloaded from Kaggle:
- **Dataset**: Face Mask Types Dataset
- **Author**: bahadoreizadkhah
- **URL**: https://www.kaggle.com/datasets/bahadoreizadkhah/face-mask-types-dataset
