from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import json
import base64

from model import Resnet50
from function import *

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
import pandas as pd
import os
from pprint import pformat
import logging
import time
from multiprocessing import cpu_count
import sys
from typing import List, Dict, Any, Optional, Tuple, Callable



app = FastAPI()

detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./noseDetection/train/result/weights/best.pt', force_reload=True) 
feature_model = EfficientNetB4(weights='imagenet', include_top=False)

@app.post("/noseDedetect")
async def noseDedetect(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = detect_model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    
    # List to store cropped image bytes
    cropped_images = []

    for result in results_json:
        # Only consider detections with confidence > 0.5
        if result['confidence'] > 0.5:
            # Extract bounding box coordinates
            xmin = result['xmin']
            ymin = result['ymin']
            xmax = result['xmax']
            ymax = result['ymax']

            # Crop the image using the coordinates
            crop_img = input_image.crop((xmin, ymin, xmax, ymax))

            # Convert the image to bytes
            byte_arr = io.BytesIO()
            crop_img.save(byte_arr, format='PNG')
            byte_img = byte_arr.getvalue()

            # Encode bytes to base64 string
            base64_bytes = base64.b64encode(byte_img)
            base64_string = base64_bytes.decode('utf-8')

            # Append to the list
            cropped_images.append(base64_string)

    # Check if any detection is made
    if cropped_images:
        return {"result": cropped_images}

    return {"result": "No object detected"}

@app.post("/dogIdent")
async def dogIdent(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = detect_model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

    for result in results_json:
        # Only consider detections with confidence > 0.5
        if result['confidence'] > 0.5:
            # Extract bounding box coordinates
            xmin = result['xmin']
            ymin = result['ymin']
            xmax = result['xmax']
            ymax = result['ymax']
            cropped = input_image.crop((xmin, ymin, xmax, ymax))

            # Convert the PIL Image to numpy array
            cropped = cropped.resize((380, 380))
            cropped_np = np.array(cropped)

            # Ensure it's in the correct format and size
            x1 = np.expand_dims(cropped_np, axis=0)
            x1 = preprocess_input(x1)

            # Process with model
            features1 = feature_model.predict(x1)
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
                features2 = feature_model.predict(x2)
                features2 = features2.flatten()
                
                # 유사도 계산
                similarity = 1 - distance.cosine(features1, features2)

                # 결과 저장
                result_df = result_df.append({'filename': filename, 'similarity': similarity}, ignore_index=True)

            # print(result_df[:10])
            # assuming df is your DataFrame and "path" is the directory containing the files
            def adjust_filename(filename):
                name, num = filename.rsplit('_', 1)
                num = num.split('.', 1)[0] # split the extension from num and take the first part
                if len(num) > 2:
                    num = num[:2]
                return f"{name}_{num}.jpg"

            result_df['filename'] = result_df['filename'].apply(adjust_filename)

            # 결과를 CSV 파일로 저장
            result_df = result_df.sort_values(by='similarity', ascending=False)
            # Get top 3 results
            top_3 = result_df.head(3)

            # Create json result
            json_result = {"dog_type": top_3.iloc[0]["filename"].split("_")[0], "top_3": []}

            for index, row in top_3.iterrows():
                json_result["top_3"].append({
                    "filename": row["filename"],
                    "similarity": row["similarity"]
                })
            
    return json_result

            