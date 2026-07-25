"""
=============================================================================
  SMOKING DETECTION IN RESTRICTED AREAS — Inference Pipeline
=============================================================================
  Model 1 : YOLOv11s (COCO pretrained) → Person Detection  [class 0]
  Model 2 : best.pt  (Custom trained)   → Cigarette Detection [class 1]
  Tracker : ByteTrack (per-person unique ID)

  Logic   : If a person's bounding box overlaps with a detected cigarette
            → Mark that person as "SMOKING" with a red alert box.
            → All other persons get a green "SAFE" box.

  Output  : Annotated video saved to VIDEO_OUT path.
=============================================================================
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
VIDEO_IN        = "input_video.mp4"
VIDEO_OUT       = "output_detected.mp4"
PERSON_MODEL    = "yolo11s.pt"          # Downloads COCO pretrained automatically
CIGARETTE_MODEL = "best.pt"             # Custom trained model
PERSON_CONF     = 0.40
CIG_CONF        = 0.25
IOU_THRESH      = 0.45
OVERLAP_THRESH  = 0.30                  # Fraction of cigarette box that must lie inside person box

# ─── Colour palette (BGR) ───────────────────
C_GREEN         = (50,  205,  50)       # Safe person
C_RED           = (30,  30,  220)       # Smoking person (BGR red)
C_ORANGE        = (0,   165, 255)       # Cigarette box
C_WHITE         = (255, 255, 255)
C_BLACK         = (0,   0,   0)
C_YELLOW        = (0,   215, 255)
C_DARK_BG       = (15,  15,  15)
C_ACCENT        = (0,   120, 255)       # Blue accent for HUD

# ─── Fonts ──────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SM         = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def iou(boxA, boxB):
    """Compute Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)


def containment(small_box, large_box):
    """Fraction of small_box area that lies inside large_box.
    Perfect for cigarette-inside-person association."""
    xA = max(small_box[0], large_box[0]);  yA = max(small_box[1], large_box[1])
    xB = min(small_box[2], large_box[2]);  yB = min(small_box[3], large_box[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area_small = ((small_box[2] - small_box[0]) *
                  (small_box[3] - small_box[1]) + 1e-6)
    return inter / area_small


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=2, filled=False):
    """Draw a rounded-corner rectangle."""
    x1, y1 = pt1;  x2, y2 = pt2
    r = min(radius, (x2-x1)//3, (y2-y1)//3)
    if filled:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)


def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.45):
    """Blend a filled rectangle with transparency."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_label(img, text, org, font=FONT, scale=0.55, color=C_WHITE,
               thickness=1, bg=None, padding=5):
    """Draw text with optional background pill."""
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if bg is not None:
        alpha_rect(img, x - padding, y - th - padding,
                   x + tw + padding, y + bl + padding, bg, alpha=0.75)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hud(frame, total_persons, smoking_count, safe_count, frame_w, frame_h):
    """Draw the live stats panel and, if smokers found, a bottom alert banner."""
    px1 = frame_w - 220
    py1 = 10
    px2 = frame_w - 8
    py2 = py1 + 132

    alpha_rect(frame, px1, py1, px2, py2, C_DARK_BG, alpha=0.75)
    draw_rounded_rect(frame, (px1, py1), (px2, py2), C_ACCENT, radius=8, thickness=1)

    cv2.putText(frame, "LIVE STATS", (px1+10, py1+22),
                FONT, 0.52, C_YELLOW, 1, cv2.LINE_AA)
    cv2.line(frame, (px1+8, py1+28), (px2-8, py1+28), C_ACCENT, 1)

    cv2.putText(frame, f"Total Persons : {total_persons}",
                (px1+10, py1+52), FONT_SM, 0.46, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SMOKING       : {smoking_count}",
                (px1+10, py1+76), FONT_SM, 0.46, C_RED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"SAFE          : {safe_count}",
                (px1+10, py1+100), FONT_SM, 0.46, C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"VIOLATIONS    : {smoking_count}",
                (px1+10, py1+124), FONT_SM, 0.46, C_ORANGE, 1, cv2.LINE_AA)

    # ── Bottom black banner (always visible, 10% height) ─────────────────
    banner_h = int(frame_h * 0.08)
    by1 = frame_h - banner_h
    cv2.rectangle(frame, (0, by1), (frame_w, frame_h), C_BLACK, -1)

    # Red "SMOKER DETECTED" text — only when violation is active
    if smoking_count > 0:
        alert_txt = "SMOKER DETECTED"
        scale = 1.1
        thick = 2
        (tw, th), _ = cv2.getTextSize(alert_txt, FONT, scale, thick)
        tx = (frame_w - tw) // 2
        ty = by1 + (banner_h + th) // 2
        cv2.putText(frame, alert_txt, (tx, ty), FONT, scale, C_RED, thick, cv2.LINE_AA)


def draw_person_box(frame, box, track_id, is_smoking, conf):
    """Draw a standard rectangle box with ByteTrack ID + confidence label."""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = C_RED if is_smoking else C_GREEN
    thick = 2

    # Plain rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # Label: ByteTrack ID + confidence  (e.g. "#3 person 0.87")
    label = f"#{track_id} person {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.50, 1)
    ly1 = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3),
                FONT_SM, 0.50, C_WHITE, 1, cv2.LINE_AA)


def draw_cigarette_box(frame, box, conf):
    """Draw a standard rectangle box for cigarette with confidence."""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = C_ORANGE
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"cigarette {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT_SM, 0.50, 1)
    ly1 = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3),
                FONT_SM, 0.50, C_WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN INFERENCE
# ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SMOKING DETECTION — LinkedIn Demo Inference")
    print("=" * 70)

    # Load models
    print("\n[1/3] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    cig_model    = YOLO(CIGARETTE_MODEL)
    print(f"      ✓ Person model   : {PERSON_MODEL}  (COCO pretrained)")
    print(f"      ✓ Cigarette model: {CIGARETTE_MODEL}  (custom trained)")

    # Open video
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_IN}")

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[2/3] Video info: {frame_w}×{frame_h} @ {fps_src:.1f}fps  "
          f"({total_fr} frames)")

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps_src, (frame_w, frame_h))

    # Running counters
    smoking_ids_history  = set()
    total_violations     = 0
    frame_idx            = 0
    proc_times           = []

    print(f"\n[3/3] Processing frames... output → {VIDEO_OUT}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # ── Person inference with ByteTrack ──────────────────────────────
        person_results = person_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],                # person only
            conf=PERSON_CONF,
            iou=IOU_THRESH,
            verbose=False
        )

        # ── Cigarette inference ───────────────────────────────────────────
        cig_results = cig_model.predict(
            frame,
            conf=CIG_CONF,
            iou=IOU_THRESH,
            verbose=False
        )

        # Parse persons
        persons = []
        if (person_results and person_results[0].boxes is not None
                and len(person_results[0].boxes) > 0):
            boxes  = person_results[0].boxes
            xyxys  = boxes.xyxy.cpu().numpy()
            confs  = boxes.conf.cpu().numpy()
            ids    = (boxes.id.cpu().numpy().astype(int)
                      if boxes.id is not None
                      else list(range(len(xyxys))))
            for xyxy, conf, tid in zip(xyxys, confs, ids):
                persons.append({"box": xyxy, "conf": conf, "id": int(tid)})

        # Parse cigarettes
        cigarettes = []
        if (cig_results and cig_results[0].boxes is not None
                and len(cig_results[0].boxes) > 0):
            for box in cig_results[0].boxes:
                cigarettes.append({
                    "box":  box.xyxy[0].cpu().numpy(),
                    "conf": float(box.conf[0].cpu())
                })

        # ── Association: which person is smoking? ─────────────────────────
        smoking_set = set()
        for cig in cigarettes:
            for p in persons:
                # Use containment: what fraction of the cigarette is inside the person box
                # Also check vanilla IoU as fallback for edge cases
                c_ratio = containment(cig["box"], p["box"])
                i_ratio = iou(cig["box"], p["box"])
                if c_ratio > OVERLAP_THRESH or i_ratio > 0.05:
                    smoking_set.add(p["id"])

        smoking_count  = len(smoking_set)
        safe_count     = len(persons) - smoking_count
        smoking_ids_history.update(smoking_set)

        # ── Draw cigarettes first (bottom layer) ─────────────────────────
        for cig in cigarettes:
            draw_cigarette_box(frame, cig["box"], cig["conf"])

        # ── Draw persons ─────────────────────────────────────────────────
        for p in persons:
            draw_person_box(frame, p["box"], p["id"],
                            p["id"] in smoking_set, p["conf"])

        # ── HUD (stats panel only) ────────────────────────────────────────
        fps_proc = 1.0 / (time.time() - t0 + 1e-6)
        proc_times.append(fps_proc)

        draw_hud(frame, len(persons), smoking_count, safe_count, frame_w, frame_h)

        out.write(frame)
        frame_idx += 1

        # Progress
        if frame_idx % 30 == 0:
            pct = frame_idx / max(total_fr, 1) * 100
            avg_fps = np.mean(proc_times[-30:])
            print(f"  Frame {frame_idx:>5}/{total_fr}  "
                  f"[{pct:5.1f}%]   avg {avg_fps:.1f} fps proc   "
                  f"smokers this frame: {smoking_count}")

    cap.release()
    out.release()

    avg_fps = np.mean(proc_times) if proc_times else 0
    print("\n" + "=" * 70)
    print("  INFERENCE COMPLETE")
    print("=" * 70)
    print(f"  Frames processed  : {frame_idx}")
    print(f"  Avg process speed : {avg_fps:.1f} fps")
    print(f"  Unique smoker IDs : {len(smoking_ids_history)}")
    print(f"\n  ✅  Output saved   → {VIDEO_OUT}")
    print("=" * 70)
    print("\n  LinkedIn-ready! Upload the output video directly.\n")


if __name__ == "__main__":
    main()
