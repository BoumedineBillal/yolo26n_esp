import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# ======================================================================================
# CONFIGURATION
# ======================================================================================

# Directory where images are located (Same dir as this script)
IMAGE_DIR = "images" 

# Model Input Size (Used for rescaling boxes to original image size)
MODEL_WIDTH = 512.0
MODEL_HEIGHT = 512.0

# Colors for bounding boxes (Matplotlib colors)
COLORS = ['r', 'b', 'g', 'c', 'm', 'y']

# ======================================================================================
# PASTE YOUR LOG HERE
# ======================================================================================
RAW_LOG_DATA = """
======================================================================
  YOLOv26n Inference Demo v3 (Refactored Processor)
======================================================================
W (1743) FbsLoader: The address of fbs model in flash is not aligned with 16 bytes.

=== Testing: bus.jpg ===
Timings:
  Pre-process:  20 ms
  Inference:    1780 ms
  Post-process: 10 ms
  Total:        1810 ms

--- Top Detections ---
Det 1: person (88.08%) | Box: [32.0, 176.0, 144.0, 432.0]
Det 2: person (81.76%) | Box: [144.0, 192.0, 224.0, 400.0]
Det 3: person (81.76%) | Box: [416.0, 176.0, 512.0, 416.0]
Det 4: bus (73.11%) | Box: [16.0, 112.0, 512.0, 368.0]
Det 5: person (50.00%) | Box: [-8.0, 264.0, 40.0, 408.0]

=== Testing: person.jpg ===
Timings:
  Pre-process:  30 ms
  Inference:    1770 ms
  Post-process: 10 ms
  Total:        1810 ms

--- Top Detections ---
Det 1: person (81.76%) | Box: [336.0, 144.0, 400.0, 416.0]
Det 2: backpack (50.00%) | Box: [340.0, 362.0, 394.0, 426.0]
Det 3: bicycle (37.75%) | Box: [192.0, 320.0, 368.0, 464.0]
Det 4: person (32.08%) | Box: [176.0, 172.0, 198.0, 206.0]
Det 5: bicycle (26.89%) | Box: [120.0, 88.0, 200.0, 168.0]

=== Test Complete ===
"""

# ======================================================================================
# PARSER LOGIC
# ======================================================================================

def parse_log_string(log_content):
    """
    Parses the ESP32 log string.
    Returns a dictionary: { "image_name.jpg": [ {class, score, x1, y1, x2, y2}, ... ] }
    """
    results = {}
    current_image = None
    
    # Regex Patterns
    img_pattern = re.compile(r"=== Testing: (.+) ===")
    det_pattern = re.compile(r"Det \d+: (.+) \(([\d.]+)%\) \| Box: \[([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+)\]")
    
    for line in log_content.strip().split('\n'):
        line = line.strip()
        
        # Check for Image Header
        img_match = img_pattern.search(line)
        if img_match:
            current_image = img_match.group(1)
            results[current_image] = []
            continue
            
        # Check for Detection Line
        if current_image:
            det_match = det_pattern.search(line)
            if det_match:
                cls_name = det_match.group(1)
                score = float(det_match.group(2))
                x1 = float(det_match.group(3))
                y1 = float(det_match.group(4))
                x2 = float(det_match.group(5))
                y2 = float(det_match.group(6))
                
                results[current_image].append({
                    "class": cls_name,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                })
    
    return results

# ======================================================================================
# VISUALIZATION LOGIC
# ======================================================================================

def visualize_results(results):
    for img_name, detections in results.items():
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            continue
            
        print(f"Processing {img_name} with {len(detections)} detections...")
        
        # Load Image
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Could not open image {img_path}: {e}")
            continue
            
        orig_width, orig_height = img.size
        
        # Calculate Scale Factors (Original / Model)
        scale_x = orig_width / MODEL_WIDTH
        scale_y = orig_height / MODEL_HEIGHT
        
        # Create Plot
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)
        
        # Draw Detections
        for i, det in enumerate(detections):
            # Rescale Coordinates
            x1 = det['box'][0] * scale_x
            y1 = det['box'][1] * scale_y
            x2 = det['box'][2] * scale_x
            y2 = det['box'][3] * scale_y
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Pick Color
            color = COLORS[i % len(COLORS)]
            
            # Create Rectangle Patch
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h, 
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add Label
            label_text = f"{det['class']} {det['score']}%"
            plt.text(
                x1, y1 - 5, label_text, 
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.5, edgecolor=color)
            )
            
        plt.axis('off')
        plt.title(f"Detections for {img_name}")
        
        plt.show()

# ======================================================================================
# MAIN
# ======================================================================================

if __name__ == "__main__":
    print(f"Parsing Raw Log Data...")
    parsed_data = parse_log_string(RAW_LOG_DATA)
    
    if not parsed_data:
        print("No detections found in log file.")
    else:
        visualize_results(parsed_data)
