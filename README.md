# Action Recognition using Vision Transformer (ViT)

This project implements action recognition on videos using the Vision Transformer (ViT) model. It includes a Streamlit-based web application for uploading videos, predicting actions, and visualizing Grad-CAM heatmaps.

---

## Features
- Train and evaluate a Vision Transformer (ViT) model for video classification.
- Web interface for uploading videos and predicting actions.
- Grad-CAM visualizations for model interpretability.

---

## Project Structure

```
action-recognition-vit
├── src
│   ├── models
│   │   └── vit.py          # Implementation of the Vision Transformer model
│   ├── training
│   │   ├── train.py        # Training script for the ViT model
│   │   └── dataset.py      # Dataset class for loading and preprocessing video data
│   ├── evaluation
│   │   └── evaluate.py     # Evaluation script for assessing model performance
│   ├── web
│   │   ├── app.py          # Web application for user interaction
│   └── utils
│       └── helpers.py      # Utility functions for data processing and visualization
├── requirements.txt         # List of project dependencies
├── README.md                # Project documentation
└── .gitignore               # Files and directories to ignore in Git
```

---

## Installation

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-repo/action-recognition-vit.git
cd action-recognition-vit
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### 1. **Training the Model**
To train the Vision Transformer model on your dataset, run the following command:
```bash
python src/training/train.py
```

### 2. **Evaluating the Model**
After training, evaluate the model's performance using:
```bash
python src/evaluation/evaluate.py
```

### 3. **Running the Web Interface**
The project includes a Streamlit-based web application for uploading videos and predicting actions.

#### Steps to Run the App:
1. Ensure the virtual environment is activated:
   ```bash
   .\.venv\Scripts\activate
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run src/web/app.py
   ```

3. Open the URL provided in the terminal (e.g., `http://localhost:8501`) in your web browser.

---

## Using the Streamlit App

### Features of the App:
- **Upload Videos**: Upload a video file in `.mp4`, `.avi`, or `.mov` format.
- **Action Prediction**: The app predicts the action in the video using the Vision Transformer model.
- **Grad-CAM Visualizations**: Visualize Grad-CAM heatmaps to understand which parts of the video influenced the model's predictions.

### Example Workflow:
1. **Upload a Video**:
   - Use the sidebar to upload a video file.
   - Supported formats: `.mp4`, `.avi`, `.mov`.

2. **View Uploaded Video**:
   - The uploaded video is displayed in the main interface.

3. **Prediction and Visualization**:
   - The app extracts frames from the video and processes them through the ViT model.
   - The predicted action is displayed, and Grad-CAM heatmaps are generated for interpretability.

4. **Interact with Results**:
   - View Grad-CAM heatmaps for each frame to understand the model's focus areas.
   - Upload another video to repeat the process.

---

## Troubleshooting

- **Dependencies Not Installed**:
  Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

- **Streamlit App Not Starting**:
  Ensure the virtual environment is activated and all dependencies are installed.

- **CUDA Issues**:
  If using a GPU, ensure PyTorch is installed with CUDA support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

---

## Screenshots

### Upload Video
![Upload Video](screenshots/upload_video.png)

### Predicted Action
![Predicted Action](screenshots/predicted_action.png)

### Grad-CAM Heatmaps
![Grad-CAM Heatmaps](screenshots/gradcam_heatmaps.png)

---

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
