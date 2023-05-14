import io
import os
import pickle

from PIL import Image

def _binary2image(binary_data) -> Image:
  image_bin = io.BytesIO(binary_data)
  image = Image.open(image_bin)
  return image

def save_pickle(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def load_pickle(path):
  if not os.path.exists(path):
    return False
  with open(path, 'rb') as f:
    return pickle.load(f)