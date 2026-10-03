"""Create a tiny dummy dataset (random-noise images) so quantization_experiment.py can run
end-to-end without real data or training. Classification metrics will be meaningless."""
import os

import numpy as np
from PIL import Image

DATA_DIR = "dataset"
CLASSES = ["class_0", "class_1"]
IMAGES_PER_CLASS = {"train": 64, "val": 32, "test": 128}   # test total = 256 images
SIZE = 64                                                    # upscaled to 224 by the transforms

rng = np.random.default_rng(42)
for split, n in IMAGES_PER_CLASS.items():
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, split, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            pixels = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(os.path.join(folder, f"{i:04d}.png"))
print(f"Dummy dataset created in '{DATA_DIR}/' (random noise, no real labels).")