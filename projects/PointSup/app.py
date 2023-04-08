import os
import numpy as np
import streamlit as st
from PIL import Image
from pyngrok import ngrok
import torch
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
from point_sup import add_point_sup_config
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def setup_model(model_weights: str, config_file: str):
    """
    Set up the detection model using the given weights and configuration file.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    add_point_sup_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    return DefaultPredictor(cfg)


def display_image(image, caption):
    """
    Display the image with the given caption.
    """
    st.image(image, caption=caption, use_column_width=True)


def bgr_to_rgb(image):
    """
    Convert an image from BGR to RGB color space.
    """
    return image[:, :, ::-1]


def get_image_from_uploaded(uploaded_image):
    """
    Convert the uploaded image from PIL format to NumPy format and from BGR to RGB color space.
    """
    input_image = Image.open(uploaded_image)
    input_image_np = np.array(input_image)
    return bgr_to_rgb(input_image_np)


def make_prediction(predictor, input_image_np):
    """
    Run the prediction on the input image using the specified predictor.
    """
    return predictor(input_image_np)


def display_prediction_result(input_image_np, predictions, meta_data):
    """
    Display the prediction result by visualizing it on the input image.
    """
    v = Visualizer(bgr_to_rgb(input_image_np), MetadataCatalog.get(meta_data))
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    display_image(bgr_to_rgb(out.get_image()), caption="推論結果")


# Define the available models and their model weights and configuration files
models = {
    "damage_detection": {
        "model_weights": "model/model_damage_final.pth",
        "config_file": "config/damage_point_coco.yaml",
        "meta_data": "custom_damage_train"
    },
    "screw": {
        "model_weights": "model/model_screw_dataset_final.pth",
        "config_file": "config/screw_point_coco.yaml",
        "meta_data": "custom_screw_train"
    }
}

# Display the title
st.title("PointSup 推論アプリ")

# Display the selection box for the models
model_name = st.selectbox("学習済みモデルを選択してください", list(models.keys()))

# Get the selected model's information
model_info = models[model_name]

# Set up the predictor with the selected model
predictor = setup_model(model_info["model_weights"], model_info["config_file"])

# Create an image upload field
uploaded_image = st.file_uploader("画像をアップロードしてください", type=["jpg", "png"])

# If an image is uploaded, display it
if uploaded_image is not None:
    input_image = Image.open(uploaded_image)
    display_image(input_image, caption="アップロードされた画像")

    # If the "Predict" button is pressed, run the prediction and display the result
    if st.button("Predict"):
        input_image_np = get_image_from_uploaded(uploaded_image)
        predictions = make_prediction(predictor, input_image_np)
        display_prediction_result(input_image_np, predictions, model_info["meta_data"])
