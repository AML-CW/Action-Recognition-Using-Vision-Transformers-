import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

def load_model():
    """
    Load the pre-trained DeiT model for image classification.
    """
    # Load the processor and model from Hugging Face
    processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

    # Modify the classifier for the number of classes (e.g., 25 classes)
    model.classifier = torch.nn.Linear(model.config.hidden_size, 25)

    # Optionally load fine-tuned weights if available
    checkpoint_path = "/content/drive/MyDrive/vitmodel/AML/src/evaluation/models/deit_model.pth"  # Update this path if you have fine-tuned weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print("Loaded fine-tuned weights from:", checkpoint_path)
    else:
        print("Using pre-trained DeiT weights.")

    return processor, model