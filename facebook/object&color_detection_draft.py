from transformers import pipeline
from datasets import load_dataset
from PIL import Image, ImageDraw
import cv2 # you should download this
import numpy as np

detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")

dataset = load_dataset("mehmetkeremturkcan/traffic-lights-of-new-york", verification_mode="no_checks")

test_image = dataset['train'][3]['image']

results = detector(
    test_image,
    candidate_labels=["traffic light"] 
)

draw = ImageDraw.Draw(test_image)

for result in results:
    if result['label'] == "traffic light" and result['score'] > 0.5:
        box = result['box']
        
        cropped_img = test_image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        
        cv_img = np.array(cropped_img)
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
        

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        if green_pixels > red_pixels and green_pixels > 10: # 최소 픽셀 수(노이즈 방지)
            current_color = "Green Light (Safe to cross)"
            outline_color = "green"
        elif red_pixels > green_pixels and red_pixels > 10:
            current_color = "Red Light (Stop)"
            outline_color = "red"
        else:
            current_color = "Unknown"
            outline_color = "gray"
            
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline=outline_color, width=4)
        print(f"Detected: Traffic Light -> {current_color}")

test_image.show()
