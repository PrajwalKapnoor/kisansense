"""Two models for the same task: a small CNN trained from scratch (baseline)
and MobileNetV2 transfer learning.

Both take raw pixels in [0, 255] and do their own rescaling, so the same
image array works everywhere (training, API, Grad-CAM).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _augmentation() -> keras.Sequential:
    # Active only during training; Keras switches these off at inference.
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_baseline(num_classes: int, image_size=(224, 224)) -> keras.Model:
    inp = keras.Input(shape=(*image_size, 3))
    x = _augmentation()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="baseline_cnn")


def build_transfer(num_classes: int, image_size=(224, 224), weights: str | None = "imagenet") -> keras.Model:
    """MobileNetV2 with a frozen backbone and a small classification head.

    The backbone and head are separate named sub-models ("backbone", "head") so
    Grad-CAM can read the last convolutional feature map.
    """
    base = keras.applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights=weights)
    backbone = keras.Model(base.input, base.output, name="backbone")
    backbone.trainable = False
    head = keras.Sequential(
        [layers.GlobalAveragePooling2D(), layers.Dropout(0.2), layers.Dense(num_classes, activation="softmax")],
        name="head",
    )
    inp = keras.Input(shape=(*image_size, 3))
    x = _augmentation()(inp)
    # Same preprocessing as keras.applications.mobilenet_v2.preprocess_input: scale to [-1, 1].
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="rescale")(x)
    x = backbone(x, training=False)  # keep BatchNorm in inference mode, also when fine-tuning
    out = head(x)
    return keras.Model(inp, out, name="mobilenetv2_transfer")


def unfreeze_top(model: keras.Model, n_layers: int = 30) -> None:
    """Prepare for fine-tuning: unfreeze the last n backbone layers (not BatchNorm)."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
