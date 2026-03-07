from imports import *
import SystemCtrl
from colour_detect import PedestrianLightClassifier
from model_wrapper import OwlViTDetector
from visualiser import draw_boxes

def main():
    detector = OwlViTDetector("./owlvit-base-patch32")

    input_folder = "testing"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    classifier = PedestrianLightClassifier()

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path)

        detections = detector.detect(image)

        final_detections = []
        for det in detections:
            label = det["label"]
            box = det["box"]
            xmin, ymin, xmax, ymax = box
            cropped = image.crop((xmin, ymin, xmax, ymax))

            w = xmax - xmin
            h = ymax - ymin
            aspect_ratio = h / w

            if aspect_ratio < 1.1:
                continue  

            overall_brightness = np.mean(np.array(cropped))

            if overall_brightness < 40:
                continue

            bright_pixels = classifier._led_brightness_score(cropped)

            if bright_pixels < 50:
                continue

            color = classifier.classify(cropped)

            if color == "green":
                det["state"] = "Green Light (Safe to cross)"
                det["outline"] = "green"
            elif color == "red":
                det["state"] = "Red Light (Stop)"
                det["outline"] = "red"
            else:
                det["state"] = "Unknown"
                det["outline"] = "gray"

            final_detections.append(det)

        annotated = draw_boxes(image, final_detections)

        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_output{ext}")
        annotated.save(output_path)

        print(f"\nProcessed: {filename}")
        for d in final_detections:
            print(f"  {d['label']} → {d['state']}")

    SystemCtrl.clear_pycache()
    sys.exit(0)

if __name__ == "__main__":
    main()