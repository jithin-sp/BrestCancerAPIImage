from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import os
import base64
from typing import Optional

app = FastAPI(title="Mammogram Analysis API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "VGG16_mammogram_model.h5"
IMG_SIZE = (150, 150)
ALPHA = 0.4
SMOOTH_SAMPLES = 3  # Reduced from 10 for faster processing
SMOOTH_NOISE = 0.03
THRESHOLD = 0.38
MIN_AREA = 300

# Global model variable
model = None
conv_layer_name = None

@app.on_event("startup")
async def load_model_on_startup():
    global model, conv_layer_name
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)  # Skip compilation for faster loading
    print("Model loaded successfully")
    
    # Find best conv layer
    conv_candidates = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    if not conv_candidates:
        raise RuntimeError("No Conv2D layers found in model")
    
    # Use last conv layer by default
    conv_layer_name = conv_candidates[-1]
    print(f"Using conv layer: {conv_layer_name}")
    
    # Warm up the model with a dummy prediction
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype('float32')
    _ = model.predict(dummy_input, verbose=0)
    print("Model warmed up and ready")

@app.get("/")
async def root():
    return {"message": "Mammogram Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def preprocess_image(image_bytes, size=IMG_SIZE):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, arr

def gradcam_plus_plus_heatmap(img_array, model, conv_layer_name, class_index):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )
        call_arg = [img_array] if isinstance(model.inputs, (list, tuple)) else img_array

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                conv_outputs, preds = grad_model(call_arg)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                score = preds[:, class_index]
            grads = tape2.gradient(score, conv_outputs)
        grads2 = tape1.gradient(grads, conv_outputs)

        conv_outputs = conv_outputs[0]
        grads = grads[0]
        grads2 = grads2[0]

        numerator = grads2
        denominator = 2.0 * grads2 + tf.reduce_sum(conv_outputs * grads, axis=(0,1), keepdims=True)
        denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator) * 1e-8)
        alpha = numerator / denominator
        alpha = tf.nn.relu(alpha)

        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        cam = tf.maximum(cam, 0)
        max_val = tf.reduce_max(cam)
        if max_val == 0 or tf.math.is_nan(max_val):
            return None
        cam = cam / (max_val + 1e-8)
        return cam.numpy()
    except Exception as e:
        print("Grad-CAM++ error:", e)
        return None

def guided_backprop(img_array, model, class_index=None):
    try:
        inp = tf.Variable(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            preds = model(inp) if not isinstance(model.inputs, (list, tuple)) else model([inp])
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            if class_index is None:
                class_index = tf.argmax(preds[0])
            score = preds[:, class_index]
        grads = tape.gradient(score, inp)[0]
        if grads is None:
            return None
        guided = tf.where(grads > 0, grads, tf.zeros_like(grads))
        gmin, gmax = tf.reduce_min(guided), tf.reduce_max(guided)
        guided = (guided - gmin) / (gmax - gmin + 1e-8)
        return guided.numpy()
    except Exception as e:
        print("Guided backprop error:", e)
        return None

def smoothgrad_gradcam(image_bytes, model, conv_layer_name, class_index, n_samples=SMOOTH_SAMPLES, noise_level=SMOOTH_NOISE):
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    base_arr = np.array(base).astype("float32") / 255.0
    cams = []
    for i in range(max(1, n_samples)):
        noise = np.random.normal(0, noise_level, base_arr.shape).astype("float32")
        noisy = np.clip(base_arr + noise, 0, 1)
        x = np.expand_dims(noisy, axis=0)
        cam = gradcam_plus_plus_heatmap(x, model, conv_layer_name, class_index)
        if cam is not None:
            cams.append(cam)
    if not cams:
        return None
    avg = np.mean(np.stack(cams, axis=0), axis=0)
    avg = np.maximum(avg, 0) / (np.max(avg) + 1e-8)
    return avg

def postprocess_heatmap_to_mask(heatmap, orig_shape, thr=THRESHOLD, min_area=MIN_AREA):
    h, w = orig_shape
    hmap_resized = cv2.resize(heatmap, (w, h))
    hmap_uint = np.uint8(255 * hmap_resized)
    _, mask = cv2.threshold(hmap_uint, int(255 * thr), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels_im = cv2.connectedComponents(mask)
    final_mask = np.zeros_like(mask)
    for lab in range(1, num_labels):
        comp = (labels_im == lab).astype("uint8") * 255
        if cv2.countNonZero(comp) >= min_area:
            final_mask = cv2.bitwise_or(final_mask, comp)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, wbox, hbox = cv2.boundingRect(contours[0])
    else:
        x, y, wbox, hbox = 0, 0, w, h
    return final_mask, (x, y, wbox, hbox)

@app.post("/analyze")
async def analyze_mammogram(file: UploadFile = File(...)):
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file is provided
    if not file:
        raise HTTPException(status_code=422, detail="No file provided. Please upload an image file.")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image file (PNG, JPG, etc.)"
        )
    
    try:
        # Read image
        print(f"[{file.filename}] Starting analysis...")
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")
        
        print(f"[{file.filename}] Image loaded: {len(image_bytes)} bytes")
        
        # Preprocess and predict
        x_batch, orig_arr = preprocess_image(image_bytes)
        preds = model.predict(x_batch, verbose=0)  # Disable progress bar
        
        if isinstance(preds, (list, tuple)):
            preds_arr = preds[0].ravel()
        else:
            preds_arr = preds[0].ravel()
        
        pred_class = int(np.argmax(preds_arr))
        confidence = float(preds_arr[pred_class])
        
        # Generate heatmap
        cam = smoothgrad_gradcam(image_bytes, model, conv_layer_name, pred_class)
        if cam is None:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap")
        
        # Guided backprop
        gb = guided_backprop(x_batch, model, class_index=pred_class)
        if gb is None:
            gb_gray = np.ones_like(cam)
        else:
            gb_gray = np.mean(gb, axis=-1)
            gb_gray = (gb_gray - gb_gray.min()) / (gb_gray.max() - gb_gray.min() + 1e-8)
        
        # Combine
        cam_resized = cv2.resize(cam, (gb_gray.shape[1], gb_gray.shape[0]))
        guided_cam = cam_resized * gb_gray
        guided_cam = guided_cam / (guided_cam.max() + 1e-8)
        
        # Post-process
        orig_full = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        mask, bbox = postprocess_heatmap_to_mask(guided_cam, orig_full.shape[:2])
        
        # Create overlay
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_full[..., ::-1], 1 - ALPHA, mask_color, ALPHA, 0)
        x, y, wbox, hbox = bbox
        cv2.rectangle(overlay, (x, y), (x + wbox, y + hbox), (0, 255, 0), 2)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', overlay)
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, mask_buffer = cv2.imencode('.png', mask)
        mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
        
        elapsed = time.time() - start_time
        print(f"[{file.filename}] Analysis complete in {elapsed:.2f}s")
        
        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": preds_arr.tolist(),
            "bbox": {"x": int(x), "y": int(y), "width": int(wbox), "height": int(hbox)},
            "overlay_image": overlay_b64,
            "mask_image": mask_b64,
            "processing_time": round(elapsed, 2)
        })
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
