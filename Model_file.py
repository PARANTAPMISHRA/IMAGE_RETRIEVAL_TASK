from transformers import pipeline
import pickle
import os
captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
captioner(r"C:\Users\pc\OneDrive\Desktop\PROJECT\DATASET_FOR_IMAGE_RETRIVAL-20240329T101610Z-001\DATASET_FOR_IMAGE_RETRIVAL\3903.jpg")
with open('/content/drive/MyDrive/Colab Notebooks/DATASET_FOR_IMAGE_RETRIVAL/searcher.pkl', 'rb') as f:
    searcher = pickle.load(f)
from PIL import Image
import matplotlib.pyplot as plt
ranked_images = searcher.rank_images("A image of Green Shrit", n=5)
path=r'/content/drive/MyDrive/Colab Notebooks/DATASET_FOR_IMAGE_RETRIVAL/'
for image in ranked_images:
  full_path=image.image_path
  relative_path=os.path.relpath(full_path, relative_path)
  image = Image.open(relative_path)
  plt.imshow(image)
  plt.axis('off')  # Turn off axis
  plt.show()