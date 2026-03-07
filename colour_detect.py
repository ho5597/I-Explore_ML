from imports import *

class PedestrianLightClassifier:
    def __init__(self, center_ratio=0.5, brightness_margin=20, led_threshold=180):
        self.center_ratio = center_ratio
        self.brightness_margin = brightness_margin
        self.led_threshold = led_threshold

    def _center_crop(self, img):
        w, h = img.size
        dw, dh = int(w * self.center_ratio), int(h * self.center_ratio)
        left = (w - dw) // 2
        top = (h - dh) // 2
        return img.crop((left, top, left + dw, top + dh))

    def _led_brightness_score(self, cropped_img):
        center = self._center_crop(cropped_img)
        gray = cv2.cvtColor(np.array(center), cv2.COLOR_RGB2GRAY)
        return np.sum(gray > self.led_threshold)

    def classify(self, cropped_img: Image.Image) -> str:
        cropped_img = self._center_crop(cropped_img)

        cv_img = np.array(cropped_img)
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)

        green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([90, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        v = hsv[:, :, 2]

        green_brightness = np.mean(v[green_mask > 0]) if np.any(green_mask) else 0
        red_brightness = np.mean(v[red_mask > 0]) if np.any(red_mask) else 0

        if green_brightness > red_brightness + self.brightness_margin:
            return "green"
        elif red_brightness > green_brightness + self.brightness_margin:
            return "red"
        else:
            return "unknown"

if __name__ == "__main__":
    pass