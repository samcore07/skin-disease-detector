import os
import json
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import timm

# Configuration & Setup
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = "skin_disease_model_final.pth"
MODEL_URL = "https://huggingface.co/samlowkey/skin-disease/resolve/main/skin_disease_model_final.pth"
CLASSES_PATH = "classes.json"
TEST_FOLDER_PATH = "dataset/val"  
IMAGE_SIZE = 224

# Download model from Hugging Face if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights from Hugging Face... Please wait a moment."):
        response = requests.get(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Dynamic Class name loading (JSON)
@st.cache_data
def load_class_names():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
    else:
        st.error(f"Missing '{CLASSES_PATH}'. Please run trainer.py first to generate the class mapping.")
        return None

CLASS_NAMES = load_class_names()

# Helper function to scan test folder recursively for images
@st.cache_data
def get_test_images(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
    image_paths = []
    if os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

# Validation transforms matching trainer.py
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Model Loading Function
@st.cache_resource
def load_model(num_classes):
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    if os.path.exists(MODEL_PATH):
        # Loading saved weights onto CPU with safe loading enabled
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    else:
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None

# 3. Streamlit Interface UI
st.title("Skin Disease Classifier")
st.write("Try on demo images or upload a clear skin image which matches the test images format, to identify potential dermatological disease\n\nUploaded images will be prioritised first, remove them to check the test cases")

st.warning("⚠️ **Disclaimer:** This is an educational diagnostic system. Do not use professionally")

if CLASS_NAMES is not None:
    model = load_model(len(CLASS_NAMES))
else:
    model = None

# Sidebar: Select from existing test images folder
st.sidebar.header("📁 Sample Test Images")
test_images = get_test_images(TEST_FOLDER_PATH)

selected_test_image = None
if test_images:
    display_names = ["-- None (Upload Your Own) --"] + [
        os.path.relpath(p, TEST_FOLDER_PATH) for p in test_images
    ]
    selection = st.sidebar.selectbox("Select a test image from folder:", display_names)
    
    if selection != "-- None (Upload Your Own) --":
        selected_index = display_names.index(selection) - 1
        selected_test_image = test_images[selected_index]
else:
    st.sidebar.info("No test image directory found or folder is empty.")

# File Upload Section (Drag & Drop)
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP, JFIF)...", type=["jpeg", "jpg", "png", "bmp", "jfif"])

# Resolve image source (Uploaded file takes priority if both are active)
image_to_process = None

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
elif selected_test_image is not None:
    image_to_process = Image.open(selected_test_image).convert("RGB")

# Running Inference if an image source is available
if image_to_process is not None and model is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        caption_text = "Uploaded Image" if uploaded_file else f"Selected: {os.path.basename(selected_test_image)}"
        st.image(image_to_process, caption=caption_text, use_container_width=True)
    
    with col2:
        input_tensor = val_transforms(image_to_process).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            top_prob, top_idx = torch.max(probabilities, 0)
            predicted_class = CLASS_NAMES[top_idx.item()]
            confidence = top_prob.item() * 100

        st.subheader("Analysis Results")
        st.success(f"**Primary Match:** {predicted_class}")
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

    st.divider()
    st.subheader("Top Predictions Breakdown")
    
    top_probs, top_indices = torch.topk(probabilities, k=min(3, len(CLASS_NAMES)))
    
    for prob, idx in zip(top_probs, top_indices):
        class_name = CLASS_NAMES[idx.item()]
        score = prob.item() * 100
        st.write(f"**{class_name}**: {score:.2f}%")
        st.progress(float(prob.item()))