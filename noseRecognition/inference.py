import os
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial import distance
import numpy as np


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

from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 미리 학습된 EfficientNet-B4 모델 불러오기
model = EfficientNetB4(weights='imagenet', include_top=False)

nose_dir = "/content/drive/MyDrive/yolov5/runs/detect/exp22/crops/nose"

# 모든 이미지 파일의 경로를 가져오는 리스트
nose_paths = [os.path.join(nose_dir, filename) for filename in os.listdir(nose_dir) if filename.endswith(('.jpg', '.jpeg', '.png'))]
# 정확도와 F1 Score를 계산하기 위한 초기값 설정
true_positive = 0
false_positive = 0
false_negative = 0

# 진행 상황을 나타내는 progress bar
pbar = tqdm(nose_paths)

# nose_paths에 있는 모든 이미지에 대해 실행
for nose_image_path in pbar:
    # 이미지 불러오기
    img1 = image.load_img(nose_image_path, target_size=(380, 380))
    x1 = image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = preprocess_input(x1)
    features1 = model.predict(x1)
    features1 = features1.flatten()
    
    # 결과를 저장할 데이터프레임 생성
    result_df = pd.DataFrame(columns=['filename', 'similarity'])

    for img2_path in nose_paths:
        if img2_path == nose_image_path: 
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
        new_row = pd.DataFrame([{'filename': os.path.basename(img2_path), 'similarity': similarity}])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # 가장 유사도가 높은 이미지 찾기
    most_similar_image = result_df.loc[result_df['similarity'].idxmax(), 'filename']

    # 이미지 파일 이름에서 id 추출
    image_id = os.path.basename(nose_image_path).split('_')[0]
    most_similar_image_id = most_similar_image.split('_')[0]
    
    # 예측이 맞았는지 확인
    if image_id == most_similar_image_id:
        true_positive += 1
    else:
        false_positive += 1

    # 진행 상황 업데이트
    pbar.set_description(f'True Positives: {true_positive}, False Positives: {false_positive}')

# 전체 이미지의 수
total_images = len(nose_paths)

# False Negative는 모든 이미지 중에서 실제로는 Positive인데 False로 예측된 경우이므로,
# 전체 이미지 수에서 True Positive를 뺀 값이 됩니다.
false_negative = total_images - true_positive

# Precision, Recall, F1 Score 계산
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1_score}')
