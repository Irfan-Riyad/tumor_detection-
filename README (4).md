
# Brain Tumor Classifier (Streamlit)

A professional Streamlit app for **44-class** brain tumor image classification with **model selection (EfficientNet-B0, DenseNet121, ResNet50)** and **explainability (Grad-CAM, LIME)**.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you need
- A **trained checkpoint** for your chosen backbone (`.pth` or `.pt`), trained on your 44 classes.
- A `classes.txt` file with **one label per line** (44 lines). See `classes_example.txt` for the format.

## Notes
- The app will still run with ImageNet weights if you don't upload a checkpoint, but predictions won't match your tumor classes.
- LIME can be slow on CPU; use it selectively.
- Grad-CAM auto-detects the last conv layer and should work across the provided backbones.
