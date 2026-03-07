# Models
from transformers import pipeline, OwlViTProcessor, OwlViTForObjectDetection

# Data handling
import numpy as np

# Image processing
from PIL import Image, ImageDraw
import cv2

# System control
import torch
import os
import shutil
import sys
from typing import List, Dict, Any

if "__main__" == __name__:
    pass