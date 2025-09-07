# YOLOv11 – Pediatric Wrist X-ray Abnormality Detection

This project uses **YOLOv11** to detect fractures and other abnormalities in pediatric wrist X-rays.  
The dataset used is **GRAZPEDWRI-DX**, organized into the YOLO format.
The model detects **9 categories of findings**:

- bone anomaly
- bone lesion
- foreign body
- fracture
- metal
- periosteal reaction
- pronator sign
- soft tissue abnormality
- text artifacts

---

## 📂 Dataset Structure

The dataset must follow this structure:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── meta.yaml
```

## 🧠 What Are Pretrained Weights?

- Files like `yolo11n.pt`, `yolo11m.pt`, `yolo11x.pt` are **pretrained weights**.
- They are trained on the **COCO dataset** (80 everyday objects: cars, people, dogs, etc.).
- Pretraining helps the model learn **basic features** (edges, textures, shapes).
- **But**: COCO doesn’t include fractures → we must **fine-tune** on a dataset.

## 🎯 Why Do We Need to Train?

- Using `yolo11m.pt` **directly** → detects COCO objects, not fractures.
- Training customizes YOLOv11 to detect the **9 medical categories**.
- After training, we get `best.pt` → a custom fracture detector.

⚡ **What this model can do:**

- Detect multiple abnormalities in **pediatric wrist X-rays only**.
- Trained on images resized to **640×640**, but inference accepts any wrist X-ray (JPEG/PNG); the model resizes internally.

📌 **What it cannot do:**

- It does **not** generalize to other bones or non-X-ray images.
- It is limited to the 9 categories defined in the dataset.

---

## 🏋️ Training Arguments Explanation

```python
# Training
model.train(
    data="/content/dataset/meta.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

**epochs=50:**

- Number of training passes over the entire dataset.
- More epochs = longer training and usually better accuracy (until overfitting).
- Typical values: 50–300 depending on dataset size.

**imgsz=640:**

- Input image size (images will be resized to this).
- Default for YOLO models is 640.
- Higher values (e.g., imgsz=1280) → more detail, but more VRAM usage.
- Lower values (e.g., imgsz=416) → faster training, less accurate.

**batch=16:**

- Number of images processed per GPU step.
- Bigger batch = faster training, but needs more VRAM.
- If you get CUDA out-of-memory errors, reduce this (e.g., batch=8 or batch=4).

### Note

When training with larger YOLOv11 models (e.g., `yolo11x.pt`, `yolo11l.pt`),  
you may notice that yolov11n.pt is downloaded (lightest model)

⚠️ This does **not** mean training is happening with the Nano model.

Ultralytics downloads `yolo11n.pt` only for **AMP (Automatic Mixed Precision) checks** to quickly test GPU compatibility.  
Your actual training still uses the specified weights (`yolo11x.pt` in this case).

---

## 📊 Training Log Explanation

```text
  Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   1/50      14.3G      1.548      2.372      1.416         73        640: 25% ━━╸───────── 222/890 0.8it/s 4:45<13:56
```

- **Epoch** → Current epoch out of total (`1/50` = 1st epoch out of 50).
- **GPU_mem** → GPU memory in use (`14.3G` = 14.3 GB on GPU).
- **box_loss** → Bounding box regression loss (lower = better localization).
- **cls_loss** → Classification loss (lower = better object class predictions).
- **dfl_loss** → Distribution Focal Loss (used for bounding box precision).
- **Instances** → Number of labeled objects in the current training batch (`73` in this example).
- **Size** → Training image size after resizing (`640`).

And the progress bar at the end shows:

- **25%** → % progress through the current epoch.
- **222/890** → Batches completed / total batches this epoch.
- **0.8 it/s** → Iterations per second (training speed).
- **4:45** → Elapsed time in this epoch.
- **<13:56** → Estimated remaining time for this epoch.

👉 Loss values should steadily **decrease** across epochs.  
If they plateau or increase, training may be done (or overfitting).

---

## 🔍 Inference on New Images

After training, use **best.pt** for predictions:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Predict on new X-ray
results = model.predict("example_wrist_xray.jpg", save=True, conf=0.25)
```

Predicted images are saved under: `runs/detect/predict/`

📦 **File Types**

- .pt → Trained PyTorch model weights (e.g., best.pt).
- .txt → YOLO bounding box annotations (used for training).
- .json → Alternative annotation format (used in YOLO training).
- .yaml → Dataset configuration file.
