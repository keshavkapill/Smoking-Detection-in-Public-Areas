# 🚭 Cigarette Violation Detection in Restricted Areas

> A real-time computer vision system that detects persons smoking in restricted/non-smoking zones using a dual-model YOLOv11 pipeline with ByteTrack multi-object tracking.

---

## 📌 Project Overview

This project implements an AI-powered surveillance system capable of:

- **Detecting persons** in a video feed using the YOLOv11s model pre-trained on the COCO dataset
- **Detecting cigarettes** using a custom-trained YOLOv11s model (`best.pt`) fine-tuned on a labelled cigarette dataset
- **Associating** cigarette detections with persons using a spatial containment algorithm
- **Tracking** each individual across frames using ByteTrack, assigning unique IDs
- **Flagging violations** — any person whose bounding box contains a cigarette detection is marked as a smoker in a restricted area

The system is designed to simulate a real-world restricted-area monitoring use case, where smoking violations can be logged, flagged, or escalated automatically.

---

## 🎯 Detection Logic

```
Video Frame
    │
    ├──► YOLOv11s (COCO)  ──► Person detections  [class 0]
    │                              │
    └──► YOLOv11s (Custom) ──► Cigarette detections [class 1]
                                   │
                     Spatial Containment Check
                     (Is cigarette inside person box?)
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
         YES → Person flagged as SMOKER           NO → Safe Person
         (Red bounding box)                       (Green bounding box)
```

**Containment Algorithm:** Rather than using standard IoU (which gives near-zero values since a cigarette box is tiny compared to a person), the system computes what fraction of the cigarette's bounding box lies within the person's bounding box. If that fraction exceeds the threshold (`0.30`), the person is flagged as smoking.

---

## 🗂️ Project Structure

```
Cigarette-Violation-Detection-In-Restricted-Areas/
│
├── Inference.py          # Main dual-model inference pipeline
├── run_inference.py      # Simple single-model inference script
├── best.pt               # Custom trained cigarette detection model
│                           (download separately — see below)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🧠 Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| `yolo11s.pt` | Person detection (class 0) | COCO pretrained — auto-downloaded by Ultralytics |
| `best.pt` | Cigarette detection (class 1) | Custom trained on labelled cigarette dataset |

### ⬇️ Download Custom Model (`best.pt`)

The custom cigarette detection model is hosted on Google Drive due to file size:

**[📥 Download best.pt from Google Drive](https://drive.google.com/file/d/1bVRSFUKuPyENUBAcZmGeOIHQh8gKLCn2/view?usp=sharing)**

Place the downloaded `best.pt` file in the root of the project directory before running inference.

---

## 🏗️ Complete Pipeline

### 1. Dataset Preparation
- Raw images collected containing cigarettes in various scenarios
- Annotations created using **CVAT** (Computer Vision Annotation Tool) in YOLO format
- Dataset cleaned: non-annotated images removed, verified bounding boxes
- Split: **80% training / 20% validation**
- Final structure exported as a YOLO-compatible dataset with `data.yaml`

### 2. Model Training
- Base model: `yolo11s.pt` (small variant, optimised for speed)
- Framework: **Ultralytics YOLOv11**
- Training performed on GPU with custom hyperparameters
- Output: `best.pt` — the best checkpoint based on validation mAP

### 3. Inference Pipeline (`Inference.py`)
- **Person model** runs `.track()` with ByteTrack enabled → persistent IDs across frames
- **Cigarette model** runs `.predict()` on the same frame
- Spatial containment check links cigarettes to persons
- Per-frame visualisation drawn using OpenCV:
  - 🟢 Green box → Safe person
  - 🔴 Red box → Smoking person
  - 🟠 Orange box → Cigarette
  - Live stats panel (top-right): Total Persons / Smoking / Safe / Violations
  - Fixed black banner (bottom 8%) → `SMOKER DETECTED` text appears in red only when a violation is active
- Output video saved as `.mp4`

---

## 📸 Detection Screenshots

<img width="1821" height="910" alt="image" src="https://github.com/user-attachments/assets/4e5ccffb-27d7-4006-9de3-034745f9ad7c" />

<br/>

<br/>


<img width="1917" height="906" alt="image" src="https://github.com/user-attachments/assets/b350e628-a539-4c65-b4d7-5966366a7d99" />

<br/>

<br/>



<img width="1888" height="912" alt="image" src="https://github.com/user-attachments/assets/6414c9bc-f918-4aa1-b65b-05512404bb4a" />



## ⚙️ Requirements

**Python:** 3.8+

Install all dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
```

> **Note:** `ultralytics` automatically handles downloading `yolo11s.pt` (COCO pretrained) on first run. You only need to manually download `best.pt`.

---

## 🚀 How to Run

### Step 1 — Clone the repository

```bash
git clone https://github.com/ADITYA-user18/Cigarette-Violation-Detection-In-Restricted-Areas.git
cd Cigarette-Violation-Detection-In-Restricted-Areas
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download the custom model

Download `best.pt` from the [Google Drive link](https://drive.google.com/file/d/1bVRSFUKuPyENUBAcZmGeOIHQh8gKLCn2/view?usp=sharing) and place it in the project root.

### Step 4 — Add your input video

Place your input video in the project root. Update the config at the top of `Inference.py`:

```python
VIDEO_IN  = "your_video.mp4"   # Input video path
VIDEO_OUT = "output.mp4"       # Output video path
```

### Step 5 — Run inference

```bash
python Inference.py
```

The processed output video will be saved in the project root.

---

## 🔧 Configuration Parameters

All parameters are adjustable in `Inference.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PERSON_CONF` | `0.40` | Minimum confidence for person detection |
| `CIG_CONF` | `0.25` | Minimum confidence for cigarette detection |
| `IOU_THRESH` | `0.45` | NMS IoU threshold |
| `OVERLAP_THRESH` | `0.30` | Containment ratio to associate cigarette with person |
| `PERSON_MODEL` | `yolo11s.pt` | COCO pretrained person detector |
| `CIGARETTE_MODEL` | `best.pt` | Custom cigarette detector |

---

## 📊 System Output — Visual Indicators

| Visual Element | Meaning |
|----------------|---------|
| 🟢 Green bounding box | Person detected — no smoking |
| 🔴 Red bounding box | Person detected — smoking in restricted area |
| 🟠 Orange bounding box | Cigarette detected |
| `ID:N` label | Unique ByteTrack ID for each person |
| `person 0.XX` | Person class + confidence score |
| `cigarette 0.XX` | Cigarette class + confidence score |
| Live Stats panel | Real-time count of persons, smokers, safe, violations |
| Bottom banner | `SMOKER DETECTED` (red) — appears only when violation is active |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11s (Ultralytics) |
| Person Detection | COCO Pretrained YOLOv11s |
| Cigarette Detection | Custom Trained YOLOv11s |
| Multi-Object Tracking | ByteTrack |
| Video Processing | OpenCV |
| Annotation Tool | CVAT |
| Language | Python 3.x |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Aditya**
- GitHub: [@ADITYA-user18](https://github.com/ADITYA-user18)
