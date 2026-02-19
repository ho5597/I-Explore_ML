
from transformers import pipeline
from datasets import load_dataset
from PIL import Image, ImageDraw

# 1. OwlViT 모델 로드 (150M Parameters)
# 이 모델은 'Zero-shot' 방식이라 찾고 싶은 단어를 함께 보내야 합니다.
detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")

# 2. 데이터셋 로드
dataset = load_dataset("mehmetkeremturkcan/traffic-lights-of-new-york", verification_mode="no_checks")

# 3. 이미지 준비
test_image = dataset['train'][3]['image']

# 4. 분석 (찾고 싶은 단어인 'candidate_labels'를 반드시 넣어야 합니다)
# 이 부분이 아까 에러가 났던 원인을 해결해줍니다.
results = detector(
    test_image,
    candidate_labels=["traffic light", "pedestrian", "car"]
)

# 5. 시각화 (신호등만 표시)
draw = ImageDraw.Draw(test_image)
print(f"\n--- Inference with 150M Parameter Model (OwlViT) ---")

for result in results:
    if result['label'] == "traffic light" and result['score'] > 0.2:
        box = result['box']
        draw.rectangle([box['xmin'], box['ymin'], box['xmax'], box['ymax']], outline="blue", width=4)
        print(f"Detected: {result['label']} | Score: {round(result['score'], 3)}")

test_image.show()
