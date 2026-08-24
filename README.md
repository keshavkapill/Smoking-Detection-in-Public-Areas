# 🚭 SMOKING DETECTION IN PUBLIC AREAS

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=210&section=header&text=🚭%20Smoking%20Detection&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38"/>
</p>

<p align="center">
  <b>AI-Powered Computer Vision System for Detecting Smoking Violations in Restricted Areas</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-Computer%20Vision-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv11-Object%20Detection-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-Video%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/keshavkapill/Smoking-Detection-in-Public-Areas?style=flat-square&color=yellow"/>
  <img src="https://img.shields.io/github/forks/keshavkapill/Smoking-Detection-in-Public-Areas?style=flat-square&color=blue"/>
  <img src="https://img.shields.io/github/last-commit/keshavkapill/Smoking-Detection-in-Public-Areas?style=flat-square&color=green"/>
</p>

---

## 🧠 What Is This Project?

**Smoking Detection in Public Areas** is an AI-powered computer vision system that detects people smoking in restricted or non-smoking areas.

Instead of simply looking for a cigarette, the system combines:

> 👤 **Person Detection** + 🚬 **Cigarette Detection** + 🎯 **Spatial Analysis** + 🆔 **Object Tracking**

This allows the system to determine whether a detected cigarette is associated with a particular person.

The project uses a **dual YOLOv11 pipeline**, **ByteTrack**, and **OpenCV** for real-time video analysis.

---

## ✨ Project At A Glance

<table>
<tr>
<td align="center" width="25%">

### 👤

**Person Detection**

Detects people using a COCO-pretrained YOLOv11 model.

</td>

<td align="center" width="25%">

### 🚬

**Cigarette Detection**

Uses a custom-trained YOLOv11 model to detect cigarettes.

</td>

<td align="center" width="25%">

### 🎯

**Smart Association**

Determines whether a cigarette belongs to a detected person.

</td>

<td align="center" width="25%">

### 🆔

**Tracking**

ByteTrack maintains unique IDs across video frames.

</td>
</tr>
</table>

---

# 🔥 How The System Works

```text
                    🎥 INPUT VIDEO
                          │
                          ▼
                 ┌─────────────────┐
                 │   Video Frame   │
                 └────────┬────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      👤 YOLOv11 COCO          🚬 Custom YOLOv11
      Person Detection         Cigarette Detection
              │                       │
              ▼                       ▼
        Person Boxes            Cigarette Boxes
              │                       │
              └───────────┬───────────┘
                          ▼
                 🎯 Spatial Analysis
                          │
                          ▼
             Is cigarette inside/within
                person's bounding box?
                          │
              ┌───────────┴───────────┐
              │                       │
             YES                     NO
              │                       │
              ▼                       ▼
        🚨 SMOKING                🟢 SAFE
          DETECTED                 PERSON
              │
              ▼
       🆔 ByteTrack ID
              │
              ▼
       📊 Live Statistics
```

The project uses spatial containment rather than standard IoU because cigarette bounding boxes are extremely small compared with person bounding boxes. A configurable containment threshold is used to associate the cigarette with a person.

---

# 🎯 Detection Logic

<table>
<tr>
<th>Detection</th>
<th>Result</th>
</tr>

<tr>
<td>👤 Person detected</td>
<td>Person bounding box created</td>
</tr>

<tr>
<td>🚬 Cigarette detected</td>
<td>Cigarette bounding box created</td>
</tr>

<tr>
<td>🎯 Cigarette inside person region</td>
<td>Potential smoking violation</td>
</tr>

<tr>
<td>🆔 ByteTrack tracking</td>
<td>Person receives a persistent ID</td>
</tr>

<tr>
<td>🚨 Violation confirmed</td>
<td>Person highlighted as smoking</td>
</tr>

</table>

---

# 🎨 Visual Detection System

<div align="center">

|    Visual Indicator    | Meaning                               |
| :--------------------: | ------------------------------------- |
|    🟢 **Green Box**    | Person detected — no smoking          |
|     🔴 **Red Box**     | Smoking person detected               |
|    🟠 **Orange Box**   | Cigarette detected                    |
|       🆔 **ID:N**      | ByteTrack tracking ID                 |
|   📊 **Stats Panel**   | Persons / Smoking / Safe / Violations |
| 🚨 **SMOKER DETECTED** | Active violation                      |

</div>

The inference pipeline draws these indicators directly on the processed video and saves the resulting output as an `.mp4` file.

---

# 🧠 AI Pipeline

<table>
<tr>
<td width="50%">

### YOLOv11s — Person Model

**Purpose**

Detect people in the video.

**Source**

COCO pretrained model.

**Output**

Person bounding boxes + confidence scores.

</td>

<td width="50%">

### YOLOv11s — Cigarette Model

**Purpose**

Detect cigarettes.

**Source**

Custom-trained model.

**Output**

Cigarette bounding boxes + confidence scores.

</td>
</tr>

<tr>
<td>

### ByteTrack

Maintains persistent identities for detected people across video frames.

</td>

<td>

### Spatial Containment

Associates a cigarette with a person using bounding-box containment.

</td>
</tr>
</table>

The repository currently uses `yolo11s.pt` for person detection and `best.pt` as the custom cigarette detector.

---

# 🏗️ Model Development Pipeline

```text
📸 DATA COLLECTION
        │
        ▼
🏷️ IMAGE ANNOTATION
      CVAT
        │
        ▼
🧹 DATA CLEANING
        │
        ▼
📚 80% TRAINING / 20% VALIDATION
        │
        ▼
🧠 YOLOv11 TRAINING
        │
        ▼
💾 best.pt
        │
        ▼
🚀 REAL-TIME INFERENCE
        │
        ▼
📊 DETECTION + TRACKING
```

The custom dataset was prepared in YOLO format, annotated using CVAT, cleaned, and split into training and validation data before model training.

---

# 🛠️ Tech Stack

<p align="center">

| Technology       | Role                             |
| ---------------- | -------------------------------- |
| 🐍 **Python**    | Core programming language        |
| 🧠 **YOLOv11**   | Object detection                 |
| 🆔 **ByteTrack** | Multi-object tracking            |
| 👁️ **OpenCV**   | Video processing & visualization |
| 🏷️ **CVAT**     | Dataset annotation               |
| 🔢 **NumPy**     | Numerical operations             |

</p>

---

# 📂 Project Structure

```text
Smoking-Detection-in-Public-Areas/
│
├── 📁 Cigarette-Violation-Detection-In-Restricted-Areas-main/
│
├── 🐍 Inference.py
│   └── Main dual-model inference pipeline
│
├── 🐍 run_inference.py
│   └── Single-model inference script
│
├── 📄 requirements.txt
│   └── Python dependencies
│
├── 📄 README.md
│
└── 🚫 best.pt
    └── Custom cigarette detection model
```

The repository currently contains the inference scripts, requirements file and README; the custom model is downloaded separately because of its file size.

---

# ⚙️ Requirements

### 🐍 Python

```text
Python 3.8+
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

# 🚀 Run The Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/keshavkapill/Smoking-Detection-in-Public-Areas.git
cd Smoking-Detection-in-Public-Areas
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Download Custom Model

Download the custom `best.pt` model from the Google Drive link provided in the repository and place it inside the project directory.

### 4️⃣ Add Your Video

Place your input video inside the project directory.

Update:

```python
VIDEO_IN = "your_video.mp4"
VIDEO_OUT = "output.mp4"
```

### 5️⃣ Start Detection

```bash
python Inference.py
```

The processed video will contain detection boxes, tracking IDs and live violation statistics.

---

# 🔧 Configuration

The inference pipeline exposes parameters that can be adjusted according to the video and detection requirements.

| Parameter         |      Default | Purpose                                |
| ----------------- | -----------: | -------------------------------------- |
| `PERSON_CONF`     |       `0.40` | Person detection confidence            |
| `CIG_CONF`        |       `0.25` | Cigarette detection confidence         |
| `IOU_THRESH`      |       `0.45` | NMS IoU threshold                      |
| `OVERLAP_THRESH`  |       `0.30` | Person–cigarette containment threshold |
| `PERSON_MODEL`    | `yolo11s.pt` | Person detector                        |
| `CIGARETTE_MODEL` |    `best.pt` | Cigarette detector                     |

These parameters are configurable in `Inference.py`.

---

# 📊 What The System Can Show

<div align="center">

### LIVE MONITORING

```text
┌─────────────────────────────────────────┐
│          📊 LIVE STATISTICS             │
│                                         │
│  👤 Total Persons     : 08              │
│  🚭 Smoking           : 02              │
│  🟢 Safe              : 06              │
│  🚨 Violations        : 02              │
│                                         │
└─────────────────────────────────────────┘
```

</div>

The actual inference visualization includes live counts for total persons, smoking persons, safe persons and violations.

---

# 🌍 Potential Applications

<table>
<tr>
<td>🏢 <b>Office Buildings</b><br><sub>Monitor designated smoke-free areas</sub></td>
<td>🏥 <b>Hospitals</b><br><sub>Support smoke-free environments</sub></td>
</tr>

<tr>
<td>🎓 <b>Educational Campuses</b><br><sub>Monitor restricted areas</sub></td>
<td>🚉 <b>Transport Hubs</b><br><sub>Assist monitoring in stations and terminals</sub></td>
</tr>

<tr>
<td>🏬 <b>Commercial Spaces</b><br><sub>Support automated monitoring</sub></td>
<td>📹 <b>Smart Surveillance</b><br><sub>Integrate computer vision into monitoring systems</sub></td>
</tr>
</table>

---

# 🚀 Future Improvements

```text
🔮 Possible Next Steps

├── 📹 CCTV / RTSP camera integration
├── 📧 Automatic email alerts
├── 📱 Mobile notifications
├── ☁️ Cloud-based violation logging
├── 📊 Web dashboard
├── 🗃️ Database-backed violation history
├── 📷 Evidence snapshot generation
└── 🏢 Multi-camera monitoring
```

---

# 💡 What This Project Demonstrates

This project brings together several important computer vision concepts:

**Object Detection**

→ Detecting people and cigarettes.

**Object Tracking**

→ Maintaining identities across frames.

**Spatial Reasoning**

→ Determining whether a cigarette belongs to a person.

**Real-Time Video Processing**

→ Processing and annotating video frames.

**AI Model Training**

→ Training a custom YOLOv11 model for cigarette detection.

**Practical AI**

→ Turning computer vision models into a real-world monitoring concept.

---

# ⭐ Why This Project Is Interesting

> **The interesting part isn't just detecting a cigarette.**

The system tries to answer a more meaningful question:

### 🚬 "Who is actually smoking?"

A cigarette detector alone can identify a cigarette.

A person detector can identify a person.

But combining:

```text
PERSON
   +
CIGARETTE
   +
SPATIAL RELATIONSHIP
   +
TRACKING
   =
🚨 POTENTIAL SMOKING VIOLATION
```

makes the system significantly more useful for real-world monitoring scenarios.

---

# 📌 Project Status

<p align="center">

<img src="https://img.shields.io/badge/Development-Completed-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Computer%20Vision-Active-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/YOLOv11-Powered-orange?style=for-the-badge"/>

</p>

---

# ⭐ Support The Project

If you found this project interesting, useful or educational:

<p align="center">

<a href="https://github.com/keshavkapill/Smoking-Detection-in-Public-Areas">
<img src="https://img.shields.io/badge/⭐%20Star%20Repository-181717?style=for-the-badge&logo=github"/>
</a>

</p>

---

# 👨‍💻 Let's Connect

<p align="center">
  <b>Interested in AI • Computer Vision • Software Development • Data • Collaboration?</b>
</p>

<p align="center">

<a href="https://github.com/KeshavKapill">
<img src="https://img.shields.io/badge/GitHub-KeshavKapill-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>

  

<a href="https://www.linkedin.com/in/keshavkapil15/">
<img src="https://img.shields.io/badge/LinkedIn-Keshav%20Kapil-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

</p>

<p align="center">

**🤝 Feel free to connect, collaborate, suggest improvements or discuss ideas!**

</p>

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=120&section=footer"/>
</p>

<p align="center">
  <sub>Built with 🧠 AI + 👁️ Computer Vision + 🐍 Python</sub>
</p>
