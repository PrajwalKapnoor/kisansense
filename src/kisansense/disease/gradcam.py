"""Grad-CAM for the MobileNetV2 transfer model."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image


def compute_gradcam(model, image: np.ndarray, class_index: int | None = None):
    """Return (heatmap in [0,1] at feature-map resolution, class probabilities, class_index).

    `image` is one HxWx3 array in [0, 255] at the model's input size. The
    heatmap shows which regions pushed the model toward `class_index`
    (default: its predicted class). It explains the model, not the plant.
    """
    rescale = model.get_layer("rescale")
    backbone = model.get_layer("backbone")
    head = model.get_layer("head")

    x = rescale(tf.convert_to_tensor(image[None], dtype=tf.float32))
    with tf.GradientTape() as tape:
        conv = backbone(x, training=False)
        tape.watch(conv)
        probs = head(conv, training=False)
        if class_index is None:
            class_index = int(tf.argmax(probs[0]))
        score = probs[:, class_index]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, probs[0].numpy(), class_index


def overlay(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend the heatmap onto the image and return an RGB uint8 array."""
    from matplotlib import colormaps

    h, w = image_uint8.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - alpha) * (image_uint8 / 255.0) + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
