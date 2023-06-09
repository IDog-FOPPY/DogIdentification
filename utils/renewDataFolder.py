import os
import shutil
from PIL import Image
import pyheif

def read_heic(img_path):
    heif_file = pyheif.read(img_path)
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_fil se.mode,
        heif_file.stride,
        )
    return image


src_dir = "/content/drive/MyDrive/DogNose/dog_image"
dst_dir = "/content/drive/MyDrive/DogDataset"

os.makedirs(dst_dir, exist_ok=True)

for folder_name in os.listdir(src_dir):
    folder_path = os.path.join(src_dir, folder_name)
    
    if os.path.isdir(folder_path):
        for i, file_name in enumerate(os.listdir(folder_path), start=1):
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.isfile(file_path):
                # 파일 확장자 체크 후, HEIC면 별도 처리
                ext = os.path.splitext(file_path)[-1].lower()
                if ext == ".heic":
                    img = read_heic(file_path)
                else:
                    img = Image.open(file_path)
                
                img = img.convert("RGB")  
                new_file_name = f"{folder_name}_{i}.jpg"
                new_file_path = os.path.join(dst_dir, new_file_name)
                img.save(new_file_path, "JPEG")