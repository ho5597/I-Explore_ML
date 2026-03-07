from imports import *

def download_model():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "owlvit-base-patch32")

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    processor.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

if "__main__" == __name__:
    pass