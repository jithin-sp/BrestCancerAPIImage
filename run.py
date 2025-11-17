# run_guided_gradcam.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ----------------- CONFIG (edit these) -----------------
MODEL_PATH = "VGG16_mammogram_model.h5"   # change if needed
IMG_PATH   =     "test3.png"              # change to your image
OUT_OVERLAY = "gradcam_guided_overlay.png"
OUT_MASK = "gradcam_mask.png"
IMG_SIZE = (150, 150)  # expected model input
ALPHA = 0.4            # overlay alpha
SMOOTH_SAMPLES = 10    # SmoothGrad samples (increase for smoother result)
SMOOTH_NOISE = 0.03    # SmoothGrad noise level
THRESHOLD = 0.38       # heatmap threshold (0-1) for mask cropping
MIN_AREA = 300         # remove tiny components under this pixel area
# ------------------------------------------------------

def load_and_print_model(path):
    print(f"Loading model from: {path}")
    m = load_model(path)
    print("\n=== Model summary (top layers) ===")
    m.summary()
    print("\n=== Layer list (index, name, type, output shape) ===")
    for i, layer in enumerate(m.layers):
        try:
            shape = getattr(layer.output, "shape", "unknown")
        except Exception:
            shape = "unknown"
        print(i, layer.name, type(layer).__name__, shape)
    print("=== end of layer list ===\n")
    return m

def find_conv_candidates(model):
    return [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

def preprocess_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    batch = np.expand_dims(arr, axis=0)
    return batch, arr

# ---------------- Grad-CAM++ (approx) ----------------
def gradcam_plus_plus_heatmap(img_array, model, conv_layer_name, class_index):
    """
    Compute Grad-CAM++ heatmap for given input batch (1,x,y,3).
    Returns (H, W) normalized heatmap or None on failure.
    """
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )
        call_arg = [img_array] if isinstance(model.inputs, (list, tuple)) else img_array

        # First and second order gradients
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                conv_outputs, preds = grad_model(call_arg)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                score = preds[:, class_index]
            grads = tape2.gradient(score, conv_outputs)   # first-order grads for gradcam++
        grads2 = tape1.gradient(grads, conv_outputs)      # second-order grads

        conv_outputs = conv_outputs[0]  # (H, W, C)
        grads = grads[0]                # (H, W, C)
        grads2 = grads2[0]              # (H, W, C)

        # alpha calculation (per Grad-CAM++ paper approximation)
        numerator = grads2
        denominator = 2.0 * grads2 + tf.reduce_sum(conv_outputs * grads, axis=(0,1), keepdims=True)
        denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator) * 1e-8)
        alpha = numerator / denominator
        alpha = tf.nn.relu(alpha)

        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0,1))  # (C,)
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)  # (H, W)
        cam = tf.maximum(cam, 0)
        max_val = tf.reduce_max(cam)
        if max_val == 0 or tf.math.is_nan(max_val):
            return None
        cam = cam / (max_val + 1e-8)
        return cam.numpy()
    except Exception as e:
        print("Grad-CAM++ error:", e)
        return None

# ---------------- Guided Backprop ----------------
def guided_backprop(img_array, model, class_index=None):
    """
    Compute guided backprop saliency (positive gradients only).
    Returns (H, W, 3) normalized to [0,1] or None.
    """
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
        grads = tape.gradient(score, inp)[0]  # (H,W,3)
        if grads is None:
            return None
        guided = tf.where(grads > 0, grads, tf.zeros_like(grads))
        gmin, gmax = tf.reduce_min(guided), tf.reduce_max(guided)
        guided = (guided - gmin) / (gmax - gmin + 1e-8)
        return guided.numpy()
    except Exception as e:
        print("Guided backprop error:", e)
        return None

# ---------------- SmoothGrad (average Grad-CAM++) ----------------
def smoothgrad_gradcam(img_path, model, conv_layer_name, class_index, n_samples=SMOOTH_SAMPLES, noise_level=SMOOTH_NOISE):
    base = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
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

# -------------- Postprocess heatmap into mask & bbox --------------
def postprocess_heatmap_to_mask(heatmap, orig_shape, thr=THRESHOLD, min_area=MIN_AREA):
    h, w = orig_shape
    hmap_resized = cv2.resize(heatmap, (w, h))
    hmap_uint = np.uint8(255 * hmap_resized)
    _, mask = cv2.threshold(hmap_uint, int(255 * thr), 255, cv2.THRESH_BINARY)
    # morphology closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # remove small components
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

# -------------- Guided Grad-CAM overlay pipeline -----------------
def guided_gradcam_pipeline(img_path, model, conv_layer_name, pred_class,
                            smooth_samples=SMOOTH_SAMPLES, smooth_noise=SMOOTH_NOISE,
                            thr=THRESHOLD, min_area=MIN_AREA, alpha=ALPHA):
    # 1) SmoothGrad-GradCAM++ heatmap
    print("Computing SmoothGrad-GradCAM++ heatmap...")
    cam = smoothgrad_gradcam(img_path, model, conv_layer_name, pred_class, n_samples=smooth_samples, noise_level=smooth_noise)
    if cam is None:
        print("Failed to compute Grad-CAM heatmap.")
        return None, None, None

    # 2) Guided backprop saliency
    print("Computing guided backprop saliency...")
    x_batch, _ = preprocess_image(img_path)
    gb = guided_backprop(x_batch, model, class_index=pred_class)
    if gb is None:
        print("Guided backprop failed; proceeding with cam-only result.")
        gb_gray = np.ones_like(cam)  # neutral
    else:
        gb_gray = np.mean(gb, axis=-1)  # collapse channels
        gb_gray = (gb_gray - gb_gray.min()) / (gb_gray.max() - gb_gray.min() + 1e-8)

    # 3) Combine cam and guided backprop (elementwise)
    cam_resized = cv2.resize(cam, (gb_gray.shape[1], gb_gray.shape[0]))
    guided_cam = cam_resized * gb_gray
    guided_cam = guided_cam / (guided_cam.max() + 1e-8)

    # 4) Post-process to mask + bbox
    orig_full = np.array(Image.open(img_path).convert("RGB"))
    mask, bbox = postprocess_heatmap_to_mask(guided_cam, orig_full.shape[:2], thr=thr, min_area=min_area)

    # 5) Overlay color mask and draw bbox
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_full[..., ::-1], 1 - alpha, mask_color, alpha, 0)
    x, y, wbox, hbox = bbox
    cv2.rectangle(overlay, (x, y), (x + wbox, y + hbox), (0, 255, 0), 2)

    return overlay, mask, bbox

# ---------------------- MAIN -----------------------
if __name__ == "__main__":
    # load model
    model = load_and_print_model(MODEL_PATH)

    # find conv candidates
    conv_candidates = find_conv_candidates(model)
    if not conv_candidates:
        raise SystemExit("No Conv2D layers found in model. Grad-CAM requires conv layers.")

    print("Conv candidates (last-first):", conv_candidates[::-1])

    # preprocess image for prediction
    x_batch, orig_arr = preprocess_image(IMG_PATH)
    preds = model.predict(x_batch)
    # flatten predictions into 1D vector
    if isinstance(preds, (list, tuple)):
        preds_arr = preds[0].ravel()
    else:
        preds_arr = preds[0].ravel()
    print("Model probabilities:", preds_arr)
    pred_class = int(np.argmax(preds_arr))
    print("Predicted class index:", pred_class)

    # try conv candidates from last -> earlier
    selected_conv = None
    for conv_name in reversed(conv_candidates):
        print("Trying conv layer:", conv_name)
        # try a single pass of gradcam++ to see if it yields a valid map
        test_cam = gradcam_plus_plus_heatmap(x_batch, model, conv_name, pred_class)
        if test_cam is not None:
            selected_conv = conv_name
            print("Selected conv layer for Grad-CAM:", conv_name)
            break
        else:
            print(" -> Grad-CAM++ returned None / empty for this layer.")

    if selected_conv is None:
        raise SystemExit("Failed to obtain a valid Grad-CAM from any conv layer. Report console output here for debugging.")

    # run guided pipeline
    overlay, mask, bbox = guided_gradcam_pipeline(IMG_PATH, model, selected_conv, pred_class,
                                                 smooth_samples=SMOOTH_SAMPLES, smooth_noise=SMOOTH_NOISE,
                                                 thr=THRESHOLD, min_area=MIN_AREA, alpha=ALPHA)
    if overlay is None:
        raise SystemExit("Guided Grad-CAM pipeline failed. See logs above.")

    # save outputs
    cv2.imwrite(OUT_OVERLAY, overlay)
    cv2.imwrite(OUT_MASK, mask)
    print("Saved overlay:", os.path.abspath(OUT_OVERLAY))
    print("Saved mask:", os.path.abspath(OUT_MASK))

    # try to open the overlay automatically on Windows
    try:
        os.startfile(os.path.abspath(OUT_OVERLAY))
    except Exception:
        pass

    # show using matplotlib as fallback
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); plt.title("Original"); plt.imshow(orig_arr); plt.axis("off")
    plt.subplot(1,2,2); plt.title("Guided Grad-CAM Overlay"); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.show()
