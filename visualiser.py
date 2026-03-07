from imports import *

def draw_boxes(image, detections):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        outline = det.get("outline", "yellow")
        label = det.get("state", det["label"])

        draw.rectangle([x1, y1, x2, y2], outline=outline, width=4)
        draw.text((x1, y1 - 12), label, fill=outline)

    return img

def crop_detection(image: Image.Image, box: List[float]) -> Image.Image:
    x1, y1, x2, y2 = map(int, box)
    return image.crop((x1, y1, x2, y2))

if __name__ == "__main__":
    pass