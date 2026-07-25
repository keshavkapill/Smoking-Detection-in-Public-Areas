import os
import shutil
from ultralytics import YOLO

model = YOLO('best.pt')
video_path = 'demo2.mp4'

# Extract the base name (e.g., 'demo4')
base_name = os.path.splitext(os.path.basename(video_path))[0]

print(f"Starting inference on {video_path}...")

# Run inference and save into a temporary folder in the current directory
# Added agnostic_nms=True to resolve overlapping classes by keeping only the highest probability detection
# iou=0.45 is the threshold for intersection over union to trigger NMS
# conf=0.3 sets a minimum probability threshold (optional but recommended)
results = model.predict(
    source=video_path, 
    save=True, 
    project='.', 
    name='temp_runs', 
    exist_ok=True, 
    show=False,
    conf=0.3,           # Minimum confidence/probability
    iou=0.45,           # NMS IOU threshold
    agnostic_nms=True   # Class-agnostic NMS (prevents multiple classes from overlapping)
)

# YOLO saves videos as .avi or .mp4 inside the output directory
source_dir = r'runs\detect\temp_runs'
output_file = None

# Find the saved video file
for file in os.listdir(source_dir):
    if file.startswith(base_name):
        output_file = os.path.join(source_dir, file)
        break

if output_file and os.path.exists(output_file):
    # Move the file directly to the root directory
    final_output = f"{base_name}_inference_output{os.path.splitext(output_file)[1]}"
    if os.path.exists(final_output):
        os.remove(final_output) # Remove if already exists
    shutil.move(output_file, final_output)
    
    # Clean up the temporary runs folder
    shutil.rmtree(source_dir)
    
    print(f"Inference complete! The output video is saved directly in the root folder as: {final_output}")
else:
    print("Error: Could not find the processed video to move.")
