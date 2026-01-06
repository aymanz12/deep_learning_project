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

```
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
```

---

---

## 🚀 Training Results

The following results were obtained during model training on the MVTec AD dataset:

| Component    | AUROC | Optimal Threshold | Training Time |
|:------------:|:-----:|:----------------:|:-------------:|
| Leather      | 0.9986| 28.48             | ~19s          |
| Metal Nut    | 0.9839| 35.02             | ~11s          |
| Bottle       | 0.9929| 34.55             | ~13s          |
| Zipper       | 0.9409| 26.60             | ~11s          |

Memory bank size: 15,000 features per component (1,536-dimensional vectors from ResNet50)

---

## 🎯 Getting Started

### 1. Prerequisites

- **Docker** (Recommended for deployment)
- **Python 3.10+** (For local development)
- **GPU** (Recommended for training; inference runs efficiently on CPU)

### 2. Generate the Models

The `.pth` model files are not included in this repository to save space. You must generate them first:

1. Open `notebook/patchcore_anomaly_detection.ipynb`
2. Run all notebook cells to:
   - Download the MVTec AD dataset
   - Train PatchCore models for each object class
   - Evaluate performance metrics
3. Trained models (e.g., `patchcore_leather.pth`) will be saved to the `models/` directory

### 3. Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open your browser to `http://127.0.0.1:8000/gradio` to use the interface.

---

## 🐳 Running with Docker

This project is containerized for seamless deployment.

### Build and Run

```bash
# Make the build script executable
chmod +x build_and_push.sh

# Build the image (replace 'v1' with your desired tag)
./build_and_push.sh v1

# Run the container
docker run -p 8000:8000 your-image-name
```

Access the app at `http://localhost:8000`

---

## 🛠️ Technology Stack

- **Algorithm**: PatchCore (ResNet50 backbone)
- **Deep Learning**: PyTorch & TorchVision
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Image Processing**: OpenCV, Pillow, NumPy
- **Containerization**: Docker

---

## 📋 Dependencies

All required packages are listed in `requirements.txt`:

```
torch
torchvision
torchaudio
fastapi
uvicorn
gradio
opencv-python
pillow
numpy
scikit-learn
scipy
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🔄 Workflow Overview

**Training Phase** (Jupyter Notebook):
1. Load MVTec AD dataset
2. Extract features using pre-trained ResNet50
3. Build memory bank from normal samples
4. Compute optimal anomaly thresholds
5. Evaluate on test set and save models

**Inference Phase** (FastAPI/Gradio):
1. Load trained model from disk
2. Extract patch-level features from input image
3. Compute nearest-neighbor distances to memory bank
4. Generate anomaly maps and classification results
5. Display results through web interface

---

## 📝 Notes

- Models are trained on 320×320 resolution images
- Memory banks contain 15,000 feature vectors per component class
- Optimal thresholds are automatically computed during training
- The system supports real-time inference with minimal latency
- All components can be easily extended to support additional object classes

---

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]
