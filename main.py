from fastapi import FastAPI, File
from PIL import Image
import io
import torch
import json
import base64

from function import *

from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance

import numpy as np

import torch
import numpy as np
from PIL import Image

import pandas as pd
import os
import boto3
from PIL import Image
import os

from typing import List
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
import json
import boto3
import uuid
from fastapi import HTTPException

app = FastAPI()

detect_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./noseDetection/train/result/weights/best.pt', force_reload=True) 
feature_model = EfficientNetB4(weights='imagenet', include_top=False)

from typing import List
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import base64
import json


def list_s3_objects(bucket_name: str, prefix: str):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return response

app = FastAPI()

@app.post("/noseDetect")
async def noseDetect(files: List[bytes] = File(...)):
    all_cropped_images = []

    for file in files:
        input_image = Image.open(io.BytesIO(file)).convert("RGB")
        results = detect_model(input_image)
        results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    
        cropped_images = []

        for result in results_json:
            if result['confidence'] > 0.4:
                xmin = result['xmin']
                ymin = result['ymin']
                xmax = result['xmax']
                ymax = result['ymax']

                crop_img = input_image.crop((xmin, ymin, xmax, ymax))

                byte_arr = io.BytesIO()
                crop_img.save(byte_arr, format='JPEG')
                byte_img = byte_arr.getvalue()

                object_name = prefix + str(uuid.uuid4()) + '.jpg'
                s3.put_object(Body=byte_img, Bucket=bucket_name, Key=object_name, ContentType='image/jpeg')

                url = f"https://{bucket_name}.s3.{s3.meta.region_name}.amazonaws.com/{object_name}"

                cropped_images.append(url)

        all_cropped_images.append(cropped_images)

    if all_cropped_images:
        return {"result": all_cropped_images}

    return {"result": "No object detected"}



@app.post("/dogIdentification")
async def dogIdent(file: bytes = File(...)):
    try:
        input_image = Image.open(io.BytesIO(file)).convert("RGB")
        results = detect_model(input_image)
        results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        
        # 결과를 저장할 데이터프레임 생성
        result_df = pd.DataFrame(columns=['filename', 'similarity'])

        for result in results_json:
            if result['confidence'] > 0.4:
                xmin = result['xmin']
                ymin = result['ymin']
                xmax = result['xmax']
                ymax = result['ymax']

                crop_img = input_image.crop((xmin, ymin, xmax, ymax))

                cropped = crop_img.resize((380, 380))
                cropped_np = np.array(cropped)

                x1 = np.expand_dims(cropped_np, axis=0)
                x1 = preprocess_input(x1)

                features1 = feature_model.predict(x1)
                features1 = features1.flatten()

                response = list_s3_objects(bucket_name, prefix)

                for obj in response['Contents']:
                    file_name = obj['Key']
                    extension = os.path.splitext(file_name)[1]
                    if extension == '.jpg':
                        print(file_name)
                        file_object = s3.get_object(Bucket=bucket_name, Key=file_name)
                        file_content = file_object["Body"].read()
                        img2 = Image.open(io.BytesIO(file_content))
                        img2 = img2.resize((380, 380))
            
                        x2 = img_to_array(img2)
                        x2 = np.expand_dims(x2, axis=0)
                        x2 = preprocess_input(x2)
                        features2 = feature_model.predict(x2)
                        features2 = features2.flatten()
                    
                        # 유사도 계산
                        similarity = 1 - distance.cosine(features1, features2)

                        # 결과 저장
                        full_url = base_url + file_name
                        result_df.loc[len(result_df)] = {'filename': full_url, 'similarity': similarity}
            
                        
        result_df = result_df.sort_values(by='similarity', ascending=False)
        return result_df.head(5).to_dict(orient="records")
    except Exception as e:
        return {"status": "error", "message": str(e)}


            