# Pedestrian Light Detection System

This project detects pedestrian traffic lights using a zero-shot OwlViT model and classifies the signal state (green, red, or unknown) using a custom brightness-based LED classifier. The system filters out car traffic lights, back-facing lights, and irrelevant detections, and outputs annotated images.

---

## Running the Project

Place your input images in the `testing/` folder, then run:

```bash
python main.py
```

or for python3:

```bash
python3 main.py
```

Annotated results will appear in the `output/` folder with filenames like:

```bash
image.jpg → image_output.jpg
```

The example images are added for simplicity.

---

## Platform Notes

```bash
sys.exit(0)
```

The script ends with `sys.exit(0)` in `main.py`.
This has only been tested on macOS.

If you see an error on Windows or Linux, simply comment out the line:

```bash
# sys.exit(0)
```

The rest of the pipeline should run normally.

## Model Troubleshooting

If anything related to the model breaks (missing weights, corrupted files, or pipeline errors), delete the entire folder:

```bash
owlvit-base-patch32/
```

Then in 'model_download.py', change the last line from 'pass' to:

```bash
download_model()
```

Then re-download the model by running:

```bash
python model_download.py
```

or for python3:

```bash
python3 model_download.py
```

This restores a clean local copy of OwlViT.

When not in use, make sure the last line is kept at `pass` to avoid unnecessary attempts of downloading the model.

---

## Dependencies
All required Python packages are listed in:

```bash
imports.py
```

Make sure your environment includes everything imported there.

---

## Project Structure
```bash
Model/
│
├── main.py
├── model_wrapper.py
├── colour_detect.py
├── visualiser.py
├── imports.py
├── model_download.py
├── SystemCtrl.py               # Intended for clearing __pycache__
│
├── owlvit-base-patch32/        # Auto-downloaded model folder
├── testing/                    # Input images
└── output/                     # Annotated results
```

---

## Notes
- The classifier uses center-cropping and brightness comparison to avoid false positives.
- The wrapper includes NMS to merge overlapping detections.
- The system is designed for still images; video support would require temporal smoothing.

