"""
FastAPI + Gradio Server for Defect Detection
"""
import os
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
from pathlib import Path
from typing import Optional
import base64
from io import BytesIO
from PIL import Image

from app.engine import PatchCoreDeploy

# Initialize FastAPI app
app = FastAPI(title="Defect Detection API", version="1.0.0")

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"

# Define available models
AVAILABLE_MODELS = {
    "leather": MODELS_DIR / "patchcore_leather.pth",
    "metalnut": MODELS_DIR / "patchcore_metalnut.pth",
    "bottle": MODELS_DIR / "patchcore_bottle.pth",
    "zipper": MODELS_DIR / "patchcore_zipper.pth",
}

# Global model instances (dictionary for dynamic loading)
models: dict[str, Optional[PatchCoreDeploy]] = {}


def load_models():
    """Load all available models at startup."""
    global models
    
    models = {}
    
    for model_name, model_path in AVAILABLE_MODELS.items():
        try:
            if model_path.exists():
                print(f"Loading {model_name} model from {model_path}...")
                models[model_name] = PatchCoreDeploy(str(model_path), model_type=model_name)
                print(f"✅ {model_name.capitalize()} model loaded successfully!")
            else:
                print(f"⚠️  {model_name.capitalize()} model not found at {model_path}")
                models[model_name] = None
        except Exception as e:
            print(f"❌ Error loading {model_name} model: {e}")
            import traceback
            traceback.print_exc()
            models[model_name] = None
    
    loaded_count = sum(1 for m in models.values() if m is not None)
    print(f"\n📊 Loaded {loaded_count}/{len(AVAILABLE_MODELS)} models successfully")


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


# Load models immediately when module is imported (before Gradio launches)
print("🔄 Pre-loading models before starting server...")
try:
    load_models()
    print("✅ Models pre-loaded successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not pre-load models: {e}")
    print("   Models will be loaded on FastAPI startup event")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict/{model_name}")
async def predict_model(model_name: str, image_data: dict):
    """
    Predict defects for images using specified model.
    """
    if model_name not in AVAILABLE_MODELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"}
        )
    
    model = models.get(model_name)
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": f"{model_name.capitalize()} model not loaded"}
        )
    
    try:
        # Decode image
        if "image" in image_data:
            # Base64 encoded image
            image_bytes = base64.b64decode(image_data["image"])
            image = np.array(Image.open(BytesIO(image_bytes)))
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No image provided"}
            )
        
        # Run prediction
        result = model.predict(image)
        
        # Convert numpy arrays to lists for JSON serialization
        if "anomaly_map" in result:
            result["anomaly_map"] = result["anomaly_map"].tolist()
        if "segmentation_mask" in result:
            result["segmentation_mask"] = result["segmentation_mask"].tolist()
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


def predict_image(image, model_type: str):
    """
    Gradio prediction function.
    """
    print(f"🔍 predict_image called: model_type={model_type}, image type={type(image)}")
    
    if image is None:
        return None, None, "Please upload an image"
    
    # Select model
    if model_type not in models:
        return None, None, f"❌ Unknown model type: {model_type}. Available: {list(models.keys())}"
    
    model = models.get(model_type)
    
    if model is None:
        model_path = AVAILABLE_MODELS.get(model_type)
        return None, None, f"❌ {model_type.capitalize()} model not loaded."
    
    try:
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Ensure RGB (remove alpha channel if present)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Ensure image is in correct format (RGB, uint8)
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
        
        # Run prediction
        print(f"🚀 Running prediction with {model_type} model...")
        result = model.predict(image_np, return_heatmap=True)
        
        # Create visualization - exact style from notebook
        if image_np.dtype == np.uint8:
            original_img = image_np.astype(np.float32) / 255.0
        else:
            original_img = image_np.astype(np.float32)
        
        # Create AI Heatmap visualization
        if result["is_defect"] and "anomaly_map" in result:
            anomaly_map = result["anomaly_map"]
            
            # Resize anomaly_map to match image size
            if anomaly_map.shape[:2] != original_img.shape[:2]:
                anomaly_map_resized = cv2.resize(anomaly_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                anomaly_map_resized = anomaly_map
            
            # Normalize heatmap 0-1
            if anomaly_map_resized.max() > anomaly_map_resized.min():
                heatmap_norm = (anomaly_map_resized - anomaly_map_resized.min()) / (anomaly_map_resized.max() - anomaly_map_resized.min())
            else:
                heatmap_norm = np.zeros_like(anomaly_map_resized)
            
            # Apply JET colormap
            heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_norm * 255), cv2.COLORMAP_JET)
            
            # Convert BGR to RGB
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            result_image = heatmap_colored
        else:
            # No defect detected - return original image in uint8
            result_image = (np.clip(original_img, 0, 1) * 255).astype(np.uint8)
        
        # Create prediction text
        status = "🔴 DEFECT DETECTED" if result["is_defect"] else "✅ NORMAL"
        score = result["score"]
        threshold = result["threshold"]
        
        prediction_text = f"""
        **{status}**
        
        **Anomaly Score:** {score:.4f}
        **Threshold:** {threshold:.4f}
        **Model:** {model_type.capitalize()}
        """
        
        # Convert original_img back to uint8 for return
        original_image_uint8 = (np.clip(original_img, 0, 1) * 255).astype(np.uint8)
        
        return original_image_uint8, result_image, prediction_text
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None, f"❌ Error: {str(e)}"


# Create Gradio interface
def create_gradio_interface():
    """Create and return Gradio interface."""
    
    with gr.Blocks(title="Defect Detection System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔍 Defect Detection System
            Upload an image to detect defects in your products.
            """
        )
        
        with gr.Row():
            with gr.Column():
                # Get available models
                available_models = [name for name, model in models.items() if model is not None]
                if not available_models:
                    available_models = list(AVAILABLE_MODELS.keys())
                
                model_choice = gr.Radio(
                    choices=available_models,
                    value=available_models[0] if available_models else "leather",
                    label="Select Model"
                )
                
                image_input = gr.Image(label="Upload Image", type="numpy", height=400)
                predict_btn = gr.Button("🔍 Detect Defects", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="Original Image", height=400)
                result_output = gr.Image(label="Result with Segmentation", height=400)
                prediction_output = gr.Markdown(label="Prediction Result")
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, model_choice],
            outputs=[image_output, result_output, prediction_output]
        )
        image_input.change(
            fn=predict_image,
            inputs=[image_input, model_choice],
            outputs=[image_output, result_output, prediction_output]
        )
    
    return demo


print("🚀 Creating Gradio interface...")
gradio_app = create_gradio_interface()

# This attaches the Gradio UI to the FastAPI app at the root ("/")
app = gr.mount_gradio_app(app, gradio_app, path="/")

print("✅ Server ready! Gradio is mounted on /")

# This block only runs if you run `python app/main.py` directly on your laptop
# Docker runs `uvicorn`, so it ignores this block (preventing the conflict)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

