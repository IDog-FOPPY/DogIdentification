import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance
import numpy as np

from model import Resnet50
from function import *

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import faiss
from natsort import natsorted

import argparse
import os
from pprint import pformat
import logging
import time
from multiprocessing import cpu_count
import sys
from typing import List, Dict, Any, Optional, Tuple, Callable


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 미리 학습된 EfficientNet-B4 모델 불러오기
model = EfficientNetB4(weights='imagenet', include_top=False)

# 이미지 불러오기
image_path='/Users/kimgyuri/foppy/DogIdentification/test/dog6_3.jpg'
nose_image_path = "/Users/kimgyuri/foppy/DogIdentification/test/nose/dog6_3.jpg"
img1 = image.load_img(nose_image_path, target_size=(380, 380))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
features1 = model.predict(x1)
features1 = features1.flatten()

# 비교할 이미지가 있는 디렉토리 경로
image_dir = "/Users/kimgyuri/foppy/DogIdentification/noseDetection/runs/detect/exp2/crops/nose"

# 결과를 저장할 데이터프레임 생성
result_df = pd.DataFrame(columns=['filename', 'similarity'])

for filename in os.listdir(image_dir):
    img2_path = os.path.join(image_dir, filename)

    # 디렉토리인 경우 건너뛰기
    if os.path.isdir(img2_path):
        continue

    # 비교할 이미지 불러오기
    img2 = image.load_img(img2_path, target_size=(380, 380))
    x2 = image.img_to_array(img2)
    x2 = np.expand_dims(x2, axis=0)
    x2 = preprocess_input(x2)
    features2 = model.predict(x2)
    features2 = features2.flatten()
    
    # 유사도 계산
    similarity = 1 - distance.cosine(features1, features2)

    # 결과 저장
    result_df = result_df.append({'filename': filename, 'similarity': similarity}, ignore_index=True)


# 가장 유사도가 높은 이미지 찾기
most_similar_image = result_df.loc[result_df['similarity'].idxmax(), 'filename']
highest_similarity = result_df['similarity'].max()

print(f'Most similar image: {most_similar_image}')
print(f'Highest similarity: {highest_similarity}')

def adjust_filename(filename):
    name, num = filename.rsplit('_', 1)
    num = num.split('.', 1)[0] # split the extension from num and take the first part
    if len(num) > 2:
        num = num[:2]
    return f"{name}_{num}.jpg"

result_df['filename'] = result_df['filename'].apply(adjust_filename)

# 결과를 CSV 파일로 저장
result_df = result_df.sort_values(by='similarity', ascending=False)
# result_df.to_csv('/content/drive/MyDrive/dog_nose/test_images/efficient_netresult_dog6.csv', index=False)
print(result_df)
# assuming df is your DataFrame and "path" is the directory containing the files
result_df['filename'] = result_df['filename'].apply(lambda x: '/Users/kimgyuri/foppy/DogIdentification/DogDataset/' + x)
result_df.to_csv('test.csv')

reference_images_dir = result_df['filename'][:10]
k_queries = 10
checkpoint_path = '/Users/kimgyuri/foppy/DogIdentification/noseRecognition/softtriple-resnet50.pth'
device: torch.device = torch.device("cpu")
print(f"Initialized device {device}")
logging.info(f"Initialized device {device}")


# Load model's checkpoint
checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
print(f"Loaded checkpoint at {checkpoint_path}")
logging.info(f"Initialized device {device}")

# Load config
config: Dict[str, Any] = checkpoint["config"]

model = Resnet50(config["embedding_size"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

batch_size: int = 48
n_workers: int = 0
n_cpus: int = cpu_count()

logging.info(f"Found {n_cpus} cpus. "
              f"Use {n_workers} threads for data loading "
              f"with batch_size={batch_size}")
print(f"Found {n_cpus} cpus. "
              f"Use {n_workers} threads for data loading "
              f"with batch_size={batch_size}")

# Initialize transform
transform = T.Compose([
    T.Resize((config["image_size"], config["image_size"])),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
logging.info(f"Initialized transforms: {transform}")
print(f"Initialized transforms: {transform}")

ALLOWED_EXTENSIONS: Tuple[str] = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
)

# Initialize reference set and reference loader
reference_set = Dataset(reference_images_dir, transform=transform)
reference_loader = DataLoader(reference_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
logging.info(f"Initialized reference loader: {reference_loader.dataset}")
print(f"Initialized reference loader: {reference_loader.dataset}")

# Calculate embeddings from images in reference set
start = time.time()
ref_embeddings, ref_labels, ref_image_paths = get_embeddings_from_dataloader(
    reference_loader, model, device
)
end = time.time()
logging.info(f"Calculated {len(ref_embeddings)} embeddings in reference set: {end - start} second")
print(f"Calculated {len(ref_embeddings)} embeddings in reference set: {end - start} second")

# Indexing database to search
index = faiss.IndexFlatL2(config["embedding_size"])
start = time.time()
index.add(ref_embeddings)
end = time.time()
logging.info(f"Indexed {index.ntotal} embeddings in reference set: {end - start} seconds")
print(f"Calculated {len(ref_embeddings)} embeddings in reference set: {end - start} second")

# Retrive k most similar images in reference set
start = time.time()
embedding: np.ndarray = get_embedding(model, image_path, transform, device)
retrieved_image_paths, retrieved_indices, retrieved_distances = query(
    index, embedding, k_queries, ref_image_paths
)
if ref_labels:
    retrieved_labels: List[str] = [ref_labels[i] for i in retrieved_indices]
end = time.time()
print(f"Done querying {k_queries} queries: {end - start} seconds")
print(f"Top {k_queries}: {pformat(retrieved_image_paths)}")

print (retrieved_distances)