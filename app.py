
import io
import os
import time
import json
import base64
import numpy as np
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Optional but recommended
import matplotlib.pyplot as plt
import cv2  # for Grad-CAM heatmap overlay
from lime import lime_image
from skimage.segmentation import mark_boundaries


# ------------------------------
# App Config
# ------------------------------
st.set_page_config(
    page_title="Brain Tumor Classifier (44 classes)",
    page_icon="🧠",
    layout="wide"
)

# Minimal CSS for a cleaner look
st.markdown(
    '''
    <style>
    .small { font-size: 0.85rem; color: #6b7280; }
    .prob { font-weight: 600; }
    .sidebar .sidebar-content { width: 340px; }
    .st-emotion-cache-1dp5vir { padding-top: 1rem; }
    </style>
    ''',
    unsafe_allow_html=True
)

AUTHOR = "Your Name" 
st.caption(f"by **{AUTHOR}**")

# ------------------------------
# Utility: Caching
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

@st.cache_resource(show_spinner=False)
def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    # Create a torchvision model and replace the classifier head for num_classes.
    arch = arch.lower()
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

@st.cache_resource(show_spinner=False)
def load_model(arch: str, num_classes: int, state_dict_bytes: Optional[bytes]) -> nn.Module:
    # Build model and load weights if provided. Accepts arbitrary state_dict keys with strict=False.
    model = build_model(arch, num_classes, pretrained=True)
    if state_dict_bytes is not None:
        try:
            buffer = io.BytesIO(state_dict_bytes)
            ckpt = torch.load(buffer, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
                # If saved with lightning or ddp, keys may have 'model.' or 'module.' prefix
                cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
            elif isinstance(ckpt, dict):
                cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in ckpt.items()}
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
            else:
                # Try raw state_dict
                missing, unexpected = model.load_state_dict(ckpt, strict=False)
            st.caption(f"Loaded checkpoint with missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
        except Exception as e:
            st.warning(f"Could not load checkpoint. Using pretrained weights. Error: {e}")
    model.eval()
    model.to(DEVICE)
    return model

@st.cache_data(show_spinner=False)
def load_class_names_from_txt(txt_bytes: bytes) -> List[str]:
    text = txt_bytes.decode("utf-8", errors="ignore")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    return names

def default_class_names(n=44) -> List[str]:
    return [f"class_{i}" for i in range(n)]

# ------------------------------
# Preprocessing & Prediction
# ------------------------------
@st.cache_resource(show_spinner=False)
def get_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(pil_img: Image.Image, image_size: int = 224) -> torch.Tensor:
    tfm = get_transform(image_size)
    tensor = tfm(pil_img).unsqueeze(0)  # shape (1, 3, H, W)
    return tensor

@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return probs.squeeze(0).detach().cpu()  # shape: (C,)

def topk(probs: torch.Tensor, k: int = 5) -> Tuple[List[int], List[float]]:
    vals, idxs = torch.topk(probs, k)
    return idxs.tolist(), vals.tolist()

# ------------------------------
# Grad-CAM
# ------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Find the last conv layer if not provided
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        self.target_layer = target_layer

        # Register hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fh = self.target_layer.register_forward_hook(forward_hook)
        self.bh = self.target_layer.register_backward_hook(backward_hook)

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        return last_conv

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor.to(DEVICE))
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # GAP over gradients
        gradients = self.gradients  # (N,C,H,W)
        activations = self.activations  # (N,C,H,W)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (N,C,1,1)
        cam = F.relu(torch.sum(weights * activations, dim=1))  # (N,H,W)
        cam = cam[0].detach().cpu().numpy()
        # Normalize to [0,1]
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

    def release(self):
        try:
            self.fh.remove()
            self.bh.remove()
        except Exception:
            pass

def overlay_cam_on_image(cam: np.ndarray, pil_img: Image.Image, alpha: float = 0.5) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)
    return Image.fromarray(overlay)

# ------------------------------
# LIME
# ------------------------------
def make_lime_explainer():
    return lime_image.LimeImageExplainer()

def _lime_predict_proba_fn(model, image_size):
    # LIME passes a batch of images as float arrays in [0,1] RGB
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def _fn(batch: np.ndarray) -> np.ndarray:
        # batch: (N,H,W,3) in [0,1]
        with torch.no_grad():
            tensors = []
            for arr in batch:
                img = Image.fromarray((arr * 255).astype(np.uint8))
                tensors.append(tfm(img))
            x = torch.stack(tensors, dim=0).to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            return probs
    return _fn

# ------------------------------
# Sidebar: Model & Assets
# ------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Choose a model, load weights, and provide class names (44 classes).")

arch = st.sidebar.selectbox(
    "Architecture",
    options=["efficientnet_b0", "densenet121", "resnet50"],
    index=0,
    help="Pick the backbone to use for inference."
)

num_classes = st.sidebar.number_input(
    "Number of classes",
    min_value=2, max_value=200, value=44, step=1
)

weights_file = st.sidebar.file_uploader(
    "Upload model weights (.pth/.pt, optional)",
    type=["pth", "pt"],
    accept_multiple_files=False
)

classes_file = st.sidebar.file_uploader(
    "Upload classes.txt (one label per line)",
    type=["txt"],
    accept_multiple_files=False
)

image_size = st.sidebar.slider("Image size", 128, 512, 224, step=32)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
show_lime = st.sidebar.checkbox("Show LIME", value=False, help="LIME is slower. Turn on when needed.")
topk_k = st.sidebar.slider("Top-K predictions to show", 1, 10, 5)

# Load class names
if classes_file is not None:
    class_names = load_class_names_from_txt(classes_file.getvalue())
    if len(class_names) != num_classes:
        st.sidebar.warning(f"classes.txt has {len(class_names)} labels but num_classes is {num_classes}. They should match.")
else:
    class_names = default_class_names(num_classes)

# Load model
state_bytes = weights_file.getvalue() if weights_file is not None else None
model = load_model(arch, num_classes, state_bytes)

# ------------------------------
# Main: Upload, Predict, Explain
# ------------------------------
st.title("🧠 Brain Tumor Classifier")
st.caption("Upload a brain MRI image. The app predicts one of the 44 classes, and can explain the decision with Grad-CAM and LIME.")

colA, colB = st.columns([1, 1])

with colA:
    uploaded = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        st.image(pil, caption="Uploaded image", use_column_width=True)
    else:
        st.info("Please upload an image to begin.")

with colB:
    st.markdown("**Model Info**")
    st.write(f"- Device: `{DEVICE}`")
    st.write(f"- Architecture: `{arch}`")
    st.write(f"- Classes: `{num_classes}`")
    if weights_file is not None:
        st.write(f"- Weights: `{weights_file.name}`")
    if classes_file is not None:
        st.write(f"- Labels: `{classes_file.name}`")
    st.caption("Tip: For best results, upload your trained checkpoint that matches the 44-class dataset and a matching classes.txt file.")

st.markdown("---")

if uploaded is not None:
    # Prediction
    with st.spinner("Running inference..."):
        x = preprocess_image(pil, image_size=image_size)
        probs = predict(model, x)
        idxs, vals = topk(probs, k=topk_k)

    # Display top-k
    st.subheader("Predictions")
    rows = []
    for rank, (i, p) in enumerate(zip(idxs, vals), start=1):
        rows.append((rank, class_names[i] if i < len(class_names) else f"class_{i}", float(p)))
    # Show as table
    st.dataframe(
        {"Rank": [r[0] for r in rows],
         "Class": [r[1] for r in rows],
         "Probability": [f"{r[2]*100:.2f}%" for r in rows]},
        hide_index=True,
        use_container_width=True
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = [r[1] for r in rows]
    values = [r[2] for r in rows]
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Top-K Predicted Probabilities")
    st.pyplot(fig, clear_figure=True)

    # Explanations in tabs
    tabs = []
    if show_gradcam and show_lime:
        tabs = st.tabs(["Grad-CAM", "LIME"])
    elif show_gradcam:
        tabs = st.tabs(["Grad-CAM"])
    elif show_lime:
        tabs = st.tabs(["LIME"])

    if show_gradcam:
        tab = tabs[0] if tabs else st.container()
        with tab:
            st.subheader("Grad-CAM")
            with st.spinner("Computing Grad-CAM..."):
                cam_util = GradCAM(model)
                cam = cam_util(x)
                cam_img = overlay_cam_on_image(cam, pil, alpha=0.5)
                cam_util.release()
            c1, c2 = st.columns(2)
            with c1:
                st.image(pil, caption="Original", use_column_width=True)
            with c2:
                st.image(cam_img, caption="Grad-CAM Overlay", use_column_width=True)
            st.caption("Grad-CAM highlights regions that most strongly influenced the predicted class.")

    if show_lime:
        tab = tabs[1] if show_gradcam and show_lime else (tabs[0] if tabs else st.container())
        with tab:
            st.subheader("LIME")
            st.caption("LIME perturbs the image and learns a simple local model to explain the prediction. This can take 10–30 seconds.")
            with st.spinner("Running LIME..."):
                explainer = make_lime_explainer()
                lime_fn = _lime_predict_proba_fn(model, image_size=image_size)
                np_img = np.array(pil).astype(np.float32) / 255.0
                # focus on top predicted class
                pred_idx = int(torch.argmax(probs).item())
                explanation = explainer.explain_instance(
                    np_img,
                    lime_fn,
                    top_labels=1,
                    hide_color=0,
                    num_samples=1000
                )
                temp, mask = explanation.get_image_and_mask(
                    label=pred_idx, positive_only=True, hide_rest=False, num_features=8, min_weight=0.0
                )
                lime_vis = mark_boundaries(temp / 255.0, mask)
                lime_vis = (lime_vis * 255).astype(np.uint8)
                st.image(lime_vis, caption=f"LIME for '{class_names[pred_idx] if pred_idx < len(class_names) else pred_idx}'", use_column_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch. Provide your own trained 44-class weights and labels to match the dataset.")
st.caption("by **Md Riyad Hossain**")
