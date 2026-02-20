import torch
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, 'data')
from model import build_model


def predict(image_path, model_path='model.pth', threshold=0.5):
    """
    Predicts attributes for a single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model weights
        threshold: Probability threshold (default: 0.5)
    
    Returns:
        list: Present attribute indices (1-4)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first: python data/train.py")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = build_model(num_classes=4, pretrained=False, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and predict
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    present_attrs = [i+1 for i, prob in enumerate(probs) if prob >= threshold]
    
    print(f"Image: {image_path}")
    print(f"Probabilities: [Attr1: {probs[0]:.3f}, Attr2: {probs[1]:.3f}, Attr3: {probs[2]:.3f}, Attr4: {probs[3]:.3f}]")
    print(f"Present attributes: {present_attrs if present_attrs else 'None'}")
    
    return present_attrs


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        predict(image_path, threshold=threshold)
    else:
        print("="*60)
        print("INFERENCE - Multilabel Image Classification")
        print("="*60)
        print("\nUsage: python inference.py <image_path> [threshold]")
        print("Example: python inference.py data/images/images/image_0.jpg 0.5\n")
        
        # Demo on sample images
        sample_images = [
            'data/images/images/image_0.jpg',
            'data/images/images/image_1.jpg',
            'data/images/images/image_2.jpg',
        ]
        
        for img_path in sample_images:
            if os.path.exists(img_path):
                print("\n" + "-"*60)
                predict(img_path)
            else:
                print(f"\nSample not found: {img_path}")
        
        print("\n" + "="*60)
