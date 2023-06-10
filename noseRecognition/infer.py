import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance
import numpy as np

from model import Resnet50
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

# 미리 학습된 EfficientNet-B4 모델 불러오기
model = EfficientNetB4(weights='imagenet', include_top=False)

# 이미지 불러오기
img1_path = "/Users/kimgyuri/foppy/DogIdentification/noseDetection/runs/detect/exp/crops/nose/dog6_2.jpg"
img1 = image.load_img(img1_path, target_size=(380, 380))
x1 = image.img_to_array(img1)
x1 = np.expand_dims(x1, axis=0)
x1 = preprocess_input(x1)
features1 = model.predict(x1)
features1 = features1.flatten()

# 비교할 이미지가 있는 디렉토리 경로
image_dir = "/Users/kimgyuri/foppy/DogIdentification/noseDetection/runs/detect/exp/crops/nose"

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

# assuming df is your DataFrame and "path" is the directory containing the files
result_df['filename'] = result_df['filename'].apply(lambda x: '/Users/kimgyuri/foppy/DogIdentification/test_images' + x)

print(result_df['filename'])


reference_images_dir = result_df['filename']
k_queries = 10
checkpoint_path = '/Users/kimgyuri/foppy/DogIdentification/noseRecognition/softtriple-resnet50.pth'

device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Initialize batch size and n_workers
batch_size: int = config.get("batch_size", None)
if not batch_size:
    batch_size = config["classes_per_batch"] * config["samples_per_class"]
n_cpus: int = cpu_count()
if n_cpus >= batch_size:
    n_workers: int = batch_size
else:
    n_workers: int = n_cpus
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


plt.rcParams['figure.figsize'] = (30, 30)
plt.rcParams['figure.dpi'] = 150


ALLOWED_EXTENSIONS: Tuple[str] = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_list: list, transform: Callable, labeled_folders: bool = False):
        self.images_list = images_list
        self.transform = transform
        self.labeled_folders: bool = labeled_folders
        self.samples: List[Tuple[str, Optional[str]]] = self.__get_all_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, Optional[str]]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def __get_all_samples(self) -> List[Tuple[str, Optional[str]]]:
        samples: List[Tuple[str, Optional[str]]] = []
        for image_path in self.images_list:
            if Dataset.has_allowed_extension(image_path):
                label: str = ""
                if self.labeled_folders:
                    label = os.path.basename(os.path.dirname(image_path))
                samples.append((image_path, label))
        return samples

    @classmethod
    def has_allowed_extension(cls, image_name: str) -> bool:
        return image_name.lower().endswith(ALLOWED_EXTENSIONS)



def get_embedding(model: torch.nn.Module,
                  image_path: str,
                  transform: Callable,
                  device: torch.device
                  ) -> np.ndarray:
    image: Image.Image = Image.open(image_path).convert("RGB")
    input_tensor: torch.Tensor = transform(image).unsqueeze(dim=0).to(device)
    embedding: torch.Tensor = model(input_tensor)
    return embedding.detach().cpu().numpy()


@torch.no_grad()
def get_embeddings_from_dataloader(loader: DataLoader,
                                   model: torch.nn.Module,
                                   device: torch.device,
                                   ) -> Tuple[np.ndarray, Optional[List[str]], List[str]]:
    model.eval()

    embeddings_ls: List[torch.Tensor] = []
    labels_ls: List[str] = []
    for images_, labels_ in loader:
        images: torch.Tensor = images_.to(device, non_blocking=True)
        embeddings: torch.Tensor = model(images)
        embeddings_ls.append(embeddings)
        labels_ls.extend(labels_)

    embeddings: torch.Tensor = torch.cat(embeddings_ls, dim=0)  # shape: [N x embedding_size]

    images_paths: List[str] = []
    for image_path, _ in loader.dataset.samples:
        images_paths.append(image_path)
    return (embeddings.cpu().numpy(), labels_ls, images_paths)


def query(index: faiss.IndexFlatL2,
          query_embedding: np.ndarray,
          k_queries: int,
          ref_image_paths: List[str],
          ) -> Tuple[List[str], List[int], List[float]]:

    # Searching using nearest neighbors in reference set
    # indices: shape [N_embeddings x k_queries]
    # distances: shape [N_embeddings x k_queries]
    distances, indices = index.search(query_embedding, k_queries)
    indices: List[int] = indices.ravel().tolist()
    distances: List[float] = distances.ravel().tolist()
    image_paths: List[str] = [ref_image_paths[i] for i in indices]
    return image_paths, indices, distances


def visualize_query(query_image_path: str,
                    retrieved_image_paths: List[str],
                    retrieved_distances: List[float],
                    retrieved_labels: Optional[List[str]] = None,
                    query_name: Optional[str] = "",
                    image_size=(224, 224)):

    n_retrieved_images: int = len(retrieved_image_paths)
    nrows: int = 2 + (n_retrieved_images - 1) // 3

    _, axs = plt.subplots(nrows=nrows, ncols=3)

    # Plot query image
    query_image: np.ndarray = cv2.imread(query_image_path, cv2.IMREAD_COLOR)
    query_image = cv2.resize(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB), image_size)
    axs[0, 1].imshow(query_image)
    title: str = "Query"
    if query_name:
        title: str = f"Query: {query_name}"
    axs[0, 1].set_title(title, fontsize=30)

    # Plot retrieved images
    for i in range(n_retrieved_images):
        row: int = i // 3 + 1
        col: int = i % 3

        retrieved_image: np.ndarray = cv2.imread(retrieved_image_paths[i], cv2.IMREAD_COLOR)
        retrieved_image = cv2.resize(cv2.cvtColor(retrieved_image, cv2.COLOR_BGR2RGB), image_size)
        distance: float = round(retrieved_distances[i], 4)
        axs[row, col].imshow(retrieved_image)

        title: str = f"Top {i + 1}\nDistance: {distance}"
        if retrieved_labels:
            label: str = retrieved_labels[i]
            title = f"Top {i + 1}: {label}\nDistance: {distance}"
        axs[row, col].set_title(title, fontsize=30)

    # Turn off axis for all plots
    for ax in axs.ravel():
        ax.axis("off")

    return axs


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

# Indexing database to search
index = faiss.IndexFlatL2(config["embedding_size"])
start = time.time()
index.add(ref_embeddings)
end = time.time()
logging.info(f"Indexed {index.ntotal} embeddings in reference set: {end - start} seconds")


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