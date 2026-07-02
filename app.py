"""
Plant Disease Classifier — Streamlit App
Run locally: streamlit run app.py
Deploy: push to GitHub, then deploy free on Hugging Face Spaces or Streamlit Community Cloud.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "plant_disease_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load model (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

model, class_names = load_model()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- UI ----------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf and I'll predict the disease (or confirm it's healthy).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)

    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    st.subheader("Prediction")
    st.write(f"**{predicted_class.replace('_', ' ')}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show top 3 predictions for transparency
    st.subheader("Top 3 predictions")
    top3_probs, top3_idx = torch.topk(probs, 3)
    for prob, idx in zip(top3_probs, top3_idx):
        st.write(f"- {class_names[idx].replace('_', ' ')}: {prob.item()*100:.2f}%")

st.markdown("---")
st.caption("Educational project — not a substitute for professional agricultural diagnosis.")
