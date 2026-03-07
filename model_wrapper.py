from imports import *

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / (area1 + area2 - inter_area)

class OwlViTDetector:
    def __init__(self, model_dir: str, device: str = None):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 0
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = -1

        print("Using device:", self.device)

        self.pipe = pipeline(
            "zero-shot-object-detection",
            model=model_dir,
            device=self.device
        )

        self.labels = [
            "a pedestrian traffic signal",
            "a pedestrian crossing light",
            "a walk signal",
            "a don't walk signal",
            "a pedestrian green man light",
            "a pedestrian red man light",
            "a crosswalk signal",
            "a pedestrian LED light"
        ]

    def nms(self, detections, iou_threshold=0.5):
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        final = []

        while detections:
            best = detections.pop(0)
            final.append(best)

            detections = [
                d for d in detections
                if iou(best["box"], d["box"]) < iou_threshold
            ]

        return final

    def detect(self, image: Image.Image, threshold: float = 0.25) -> list[dict]:
        results = self.pipe(
            image,
            candidate_labels=self.labels,
            threshold=threshold
        )

        detections = []
        for det in results:
            box = det["box"]
            detections.append({
                "label": det["label"],
                "score": float(det["score"]),
                "box": [
                    float(box["xmin"]),
                    float(box["ymin"]),
                    float(box["xmax"]),
                    float(box["ymax"])
                ]
            })

        detections = self.nms(detections, iou_threshold=0.5)

        return detections


if __name__ == "__main__":
    pass