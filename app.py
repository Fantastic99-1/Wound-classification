"""WoundNet demo: upload a wound photo and get the predicted wound type.

Run:  python app.py
Then open the local link it prints (add share=True to launch() for a public link).
"""
import numpy as np
import tensorflow as tf
import gradio as gr

MODEL_PATH = "models/EfficientNetB0_0hidden.keras"
CLASS_NAMES = ["Background", "Diabetic", "Normal", "Pressure", "Surgical", "Venous"]
IMG_SIZE = (224, 224)

# Load once at startup. The model contains its own preprocessing,
# so it takes raw 0-255 RGB pixels, exactly like during training.
model = tf.keras.models.load_model(MODEL_PATH)


def predict(image):
    x = tf.image.resize(image.astype("float32"), IMG_SIZE)  # same resize as training
    probs = model.predict(x[tf.newaxis, ...], verbose=0)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Wound photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="WoundNet - wound type classifier",
    description=(
        "Classifies a cropped wound photo into Background, Diabetic, Normal, Pressure, "
        "Surgical or Venous (EfficientNetB0, ~80% test accuracy). "
        "Student project for learning purposes - not a medical device."
    ),
)

if __name__ == "__main__":
    demo.launch()
