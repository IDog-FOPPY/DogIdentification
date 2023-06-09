from glob import glob
from sklearn.model_selection import train_test_split
import yaml
import os
import xml.etree.ElementTree as ET

def makeYaml():
    img_path = '/content/drive/MyDrive/dog_nose/images'
    dataset_path = '/content/drive/MyDrive/dog_nose/dataset'
    img_list = glob(img_path + '/*.png')
    train_img_list, val_img_list = train_test_split(img_list, test_size = 0.1, random_state = 2000)

    with open(dataset_path+'/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
    with open(dataset_path+'/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')


    with open(dataset_path+'/data.yaml', 'r') as f:
    data = yaml.safe_load(f)

    data['train'] = dataset_path+'/train.txt'
    data['val'] = dataset_path+'/val.txt'

    with open(dataset_path+'/data.yaml', 'w') as f:
    yaml.dump(data, f)

def annotation():
    xml_dir_path = '/content/drive/MyDrive/nose/annotations'
    output_dir_path = '/content/drive/MyDrive/dog_nose/labels'

    xml_files = glob(os.path.join(xml_dir_path, '*.xml'))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            class_id = 0

            filename = os.path.basename(xml_file)
            filename_without_ext = os.path.splitext(filename)[0]
            txt_file_path = os.path.join(output_dir_path, filename_without_ext + '.txt')
            with open(txt_file_path, 'a') as f:
                f.write(f'{class_id} {x_center} {y_center} {w} {h}\n')

if __name__ == '__main__':
    makeYaml()
    annotation()