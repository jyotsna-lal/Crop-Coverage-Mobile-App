# export/export_to_tflite.py
import tensorflow as tf

# --- custom layer ---
class PixelShuffle2D(tf.keras.layers.Layer):
    def __init__(self, upsampling_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.upsampling_factor = upsampling_factor
    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upsampling_factor)
    def get_config(self):
        config = super().get_config()
        config.update({"upsampling_factor": self.upsampling_factor})
        return config

# --- custom metrics ---
def multiclass_dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    intersection = tf.reduce_sum(y_true * y_pred, axis=1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=1)
    return tf.reduce_mean((2. * intersection + smooth) / (denominator + smooth))

def combined_loss(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1 - 1e-7)
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    dice = 1 - multiclass_dice_coef(y_true, y_pred)
    return cce + dice

# --- load model ---
model = tf.keras.models.load_model(
    "../models/best_transformer_model.h5",  # path to your trained transformer
    compile=False,
    custom_objects={
        "PixelShuffle2D": PixelShuffle2D,
        "multiclass_dice_coef": multiclass_dice_coef,
        "combined_loss": combined_loss
    }
)

# --- export to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optimize for size/speed
tflite_model = converter.convert()

# --- save model ---
with open("../models/transformer_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Export complete: transformer_model.tflite saved in models/")
