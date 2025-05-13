# This file visualise the model on the testing set using GradCam and save the heatmaps.
# Process the first clip in the batch for visualization

import sys
import os
import logging
import warnings

# Add the src directory to sys.path
sys.path.append('/content/drive/MyDrive/ColabNotebooks/AML_Coursework/01_AML_Dir/src')

from models.timesformer import load_timesformer_model
from training.dataset import get_dataloader

import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

warnings.filterwarnings('ignore')

# Monkey-patch GradCAM to handle tuple outputs (for attention layers, not needed for patch embedding but kept for robustness)
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def patched_save_activation(self, module, input, output):
    activation = output
    if isinstance(activation, tuple):
        activation = activation[0]
    self.activations.append(activation.cpu().detach())
ActivationsAndGradients.save_activation = patched_save_activation

# Wrapper so GradCAM gets only logits from model
class TimeSformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).logits

# Configure logging
logging.basicConfig(filename='visualisation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_dir, batch_size=8, model_path="timesformer_model.pth", output_dir="gradcam_outputs"):
    """
    Visualise the model on the testing set using GradCam and save the heatmaps.
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        model_path (str): Path to the trained model weights.
        output_dir (str): Directory to save Grad-CAM heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model and move to device
    try:
        processor, model = load_timesformer_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        logging.info(f"Loaded model weights from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file {model_path} not found. Using pre-trained model without weights.")
        processor, model = load_timesformer_model(num_labels=25)
        model.to(device)
    model.eval()

    # Wrap model so GradCAM gets only logits
    wrapped_model = TimeSformerWrapper(model)

    # Load data (returns train, val, test loaders, using only test loader)
    _, _, test_loader = get_dataloader(data_dir, batch_size=batch_size, clip_size=8, train_ratio=0.8, val_ratio=0.1)
    logging.info(f"Test loader size: {len(test_loader.dataset)} samples")
    dataloader = test_loader

    # Set up Grad-CAM
    # Use the patch embedding projection layer as the target
    target_layers = [model.timesformer.embeddings.patch_embeddings.projection]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers, use_cuda=(device.type == 'cuda'))

    for idx, batch in enumerate(dataloader):
        video_clip, label = batch

        # Print shape for debugging
        print("video_clip shape before permute:", video_clip.shape)

        # If input is [batch, 3, 8, 224, 224], permute to [batch, 8, 3, 224, 224]
        if video_clip.shape[1] == 3 and video_clip.shape[2] == 8:
            video_clip = video_clip.permute(0, 2, 1, 3, 4)
        video_clip = video_clip.to(device)
        label = label.to(device)

        # Process only the first clip in the batch for visualization
        clip = video_clip[0].unsqueeze(0)  # (1, 8, 3, 224, 224)

        # Forward pass to get predicted class
        with torch.no_grad():
            logits = wrapped_model(clip)
        predicted_class = torch.argmax(logits, dim=1).item()

        # Grad-CAM
        grayscale_cam = cam(input_tensor=clip, targets=[ClassifierOutputTarget(predicted_class)])
        # grayscale_cam shape: (1, 8, 14, 14) for patch embedding (if patch size 16 and image size 224)
        # We'll visualize the first frame (index 0)
        heatmap = grayscale_cam[0, 0]  # (14, 14)

        # Upsample heatmap to 224x224 for overlay
        import cv2
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Prepare the frame for overlay (first frame, first channel)
        frame = clip[0, 0].detach().cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
        frame -= frame.min()
        if frame.max() != 0:
            frame /= frame.max()

        overlay = show_cam_on_image(frame, heatmap_resized, use_rgb=True)
        overlay_image = Image.fromarray(overlay)
        filename = os.path.join(output_dir, f"gradcam_heatmap_{idx}.png")
        overlay_image.save(filename)
        logging.info(f"Saved Grad-CAM heatmap: {filename}")

        # Display
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.title(f"Grad-CAM Heatmap (Sample {idx})")
        plt.show()

if __name__ == "__main__":
    # Example usage; adjust paths as needed
    DATA_DIR = "/content/drive/MyDrive/ColabNotebooks/AML_Coursework/HMDB_simp"
    MODEL_PATH = "timesformer_model.pth"
    OUTPUT_DIR = "gradcam_outputs"
    evaluate_model(DATA_DIR, batch_size=8, model_path=MODEL_PATH, output_dir=OUTPUT_DIR)
