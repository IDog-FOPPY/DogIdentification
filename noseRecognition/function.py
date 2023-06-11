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

ALLOWED_EXTENSIONS: Tuple[str] = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
)