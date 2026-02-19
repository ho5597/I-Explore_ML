from transformers import pipeline
from datasets import load_dataset
from PIL import Image, ImageDraw

# 1. Initialize the Object Detection Pipeline
detector = pipeline("object-detection", model="facebook/detr-resnet-101")

# 2. Load the Dataset
dataset = load_dataset("mehmetkeremturkcan/traffic-lights-of-new-york", verification_mode="no_checks")

# 3. Get a valid image
test_image = dataset['train'][3]['image']
results = detector(test_image)

# 4. Prepare to draw on the image
draw = ImageDraw.Draw(test_image)

print("\n--- Visualizing Traffic Light Detections Only ---")
for result in results:
    box = result['box']
    label = result['label']
    score = result['score']

    # --- 수정된 부분: label이 'traffic light'이고 확신도가 0.5(50%) 이상인 경우만 그림 ---
    if label == "traffic light" and score > 0.90:
        # Extract coordinates
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        
        # Draw the bounding box (신호등이니까 눈에 띄게 초록색으로 그려볼까요?)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)
        
        # Draw the label and score
        draw.text((xmin, ymin - 15), f"{label}: {round(score,2)}", fill="green")
        
        print(f"Traffic light found at ({xmin}, {ymin}) with confidence {round(score,2)}")

# 5. Show the final image
test_image.show()
