# 🔍 PatchCore Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

A high-performance industrial anomaly detection system based on the **PatchCore** algorithm. This project uses deep learning to detect defects in manufacturing components (e.g., Leather, Bottles, Metal Nuts, Zippers) with near-perfect accuracy.

The system includes a training pipeline (Jupyter Notebook) and a deployment-ready web application powered by **FastAPI** and **Gradio**.

---

## 📸 App Demo

![App Interface](assets/app_demo_screenshot.png)
*Figure 1: The Gradio web interface detecting anomalies in real-time.*

---

## 📂 Project Structure

```text
.
├── app/
│   ├── engine.py          # Core inference logic (PatchCore implementation)
│   ├── main.py            # FastAPI & Gradio application entry point
│   └── __init__.py
├── models/                # Directory for saved .pth model files
├── notebook/
│   └── patchcore_anomaly_detection.ipynb  # Training & Evaluation pipeline
├── build_and_push.sh      # Docker build script
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── README.md

📊 Evaluation Results

The model was evaluated on the MVTec AD dataset (320x320 resolution) with the following performance metrics:
OBJECT CLASS	IMG AUROC	PIX AUROC	F1 SCORE	ACCURACY	PRECISION	RECALL
LEATHER	1.0000	0.9912	1.0000	1.0000	1.0000	1.0000
BOTTLE	1.0000	0.9818	1.0000	1.0000	1.0000	1.0000
METAL NUT	0.9873	0.9514	0.9838	0.9739	0.9891	0.9785
ZIPPER	0.9307	0.9647	0.9442	0.9139	0.9649	0.9244
SYSTEM MEAN	0.9795	0.9723	0.9820	0.9720	0.9885	0.9757
🚀 Getting Started
1. Prerequisites

    Docker (Recommended for deployment)

    Python 3.10+ (For local development)

    GPU (Recommended for training, but inference runs on CPU)

2. Generate the Models

Note: The .pth model files are not included in this repository to save space. You must generate them first.

    Open notebook/patchcore_anomaly_detection.ipynb.

    Run the notebook cells to download the MVTec dataset and train the models.

    The trained models (e.g., patchcore_leather.pth) will be saved into the models/ directory.

3. Run Locally

Install the dependencies:
Bash

pip install -r requirements.txt

Start the FastAPI server:
Bash

uvicorn app.main:app --reload

Open your browser to http://127.0.0.1:8000/gradio to use the interface.
🐳 Running with Docker

This project is containerized for easy deployment.

Build and Run:
Bash

# Make the build script executable
chmod +x build_and_push.sh

# Build the image (replace 'your-tag' with a name, e.g., 'v1')
./build_and_push.sh v1

# Run the container
docker run -p 8000:8000 your-image-name

Access the app at http://localhost:8000.
🛠️ Technology Stack

    Algorithm: PatchCore (ResNet50 Backbone)

    Framework: PyTorch & TorchVision

    Backend: FastAPI (Python)

    Frontend: Gradio

    Image Processing: OpenCV, Pillow, NumPy
