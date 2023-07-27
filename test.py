import boto3
from PIL import Image
import os
import io

s3 = boto3.client('s3', region_name='ap-northeast-2', aws_access_key_id='AKIA6KAK66LSLKXZISSY', aws_secret_access_key='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')

bucket_name = 'foppy'
prefix = 'nose/'

response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

for obj in response['Contents']:
    file_name = obj['Key']
    extension = os.path.splitext(file_name)[1]
    if extension == '.jpg':
        print(file_name)
        file_object = s3.get_object(Bucket=bucket_name, Key=file_name)
        file_content = file_object["Body"].read()
        image = Image.open(io.BytesIO(file_content))
        image.show()