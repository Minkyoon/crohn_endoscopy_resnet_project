import random
import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import pandas as pd
from torch.utils.data import Dataset

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용하도록)


from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 모델 설정 값


config = {
    # Classfier 설정
    "cls_hidden_dims" : [1024, 512, 256]
    }



class CovidResNet(nn.Module):
    """pretrain 된 ResNet을 이
    """
    
    def __init__(self):
        """
		Args:
			base_model : resnet18 / resnet50
			config: 모델 설정 값
		"""
        super(CovidResNet, self).__init__()
       
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        self.num_ftrs = num_ftrs
        
        
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False

            

        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)

        return x





model_image = CovidResNet()
model_image

class Classifier(nn.Sequential):
    """임베딩 된 feature를 이용해 classificaion
    """
    def __init__(self, model_image, **config):
        """
        Args:
            model_image : image emedding 모델
            config: 모델 설정 값
        """
        super(Classifier, self).__init__()

        self.model_image = model_image # image 임베딩 모델

        self.input_dim = model_image.num_ftrs # image feature 사이즈
        self.dropout = nn.Dropout(0.5) # dropout 적용

        self.hidden_dims = config['cls_hidden_dims'] # classifier hidden dimensions
        layer_size = len(self.hidden_dims) + 1 # hidden layer 개수
        dims = [self.input_dim] + self.hidden_dims + [2] 

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)]) # classifer layers 

    def forward(self, v):
        # Drug/protein 임베딩
        v_i = self.model_image(v) # batch_size x hidden_dim 

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer,
                v_i = l(v_i)
            else:
                # If Not last layer, dropout과 ReLU 적용
                v_i = F.relu(self.dropout(l(v_i)))

        return v_i



device = torch.device("cuda:0" )

model = torch.load('/home/minkyoon/crom/pyfile/relapse/resnet50_epoch100_trans4.pt')


fc_layer_list = list(model.predictor.children())
model.predictor = nn.Sequential(*fc_layer_list[:1])


model=model.to(device)


def extract_features(model, dataloader):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Do not calculate gradients
        feature_list = []
        for img, label in dataloader:
            img = img.to(device) # Move the image tensor to GPU
            features = model(img) # Forward pass to get the features
            # features shape is [batch_size, 1024]
            feature_list.append(features)
        # Concatenate all feature tensors along the batch dimension
        features_all = torch.cat(feature_list, dim=0)
    return features_all

# You can create a DataLoader for images of a certain accession number
# Assume that 'custom_dataset' is a dataset for images of that accession number



from torch.utils.data import Dataset
import numpy as np
import pandas as pd


transform = transforms.Compose([
    transforms.ToTensor(), 
           
    
])

# 사용자 정의 Dataset 클래스


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSV 파일 로딩
        self.data_info = pd.read_csv(csv_file)
        self.transform=transform

        # 이미지와 레이블 데이터의 열 이름을 알고 있어야 합니다.
        self.image_arr = self.data_info['filepath']
        self.label_arr = self.data_info['label']
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # 이미지 파일을 numpy 배열로 불러오기
        single_image_name = self.image_arr[index]
        image = np.load(single_image_name).astype(np.float32)
        # numpy 배열을 PyTorch 텐서로 변환
        if isinstance(image, tuple):
            image = image[0]
            
        
        
        
        if self.transform:
           
            
            image_as_tensor = self.transform(image)
             

        # 레이블 불러오기 (레이블도 PyTorch 텐서로 변환할 수 있습니다)
        single_image_label = self.label_arr[index]
        
        return (image_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
    
import os

# CSV 파일이 있는 디렉토리 경로
csv_directory = "/home/minkyoon/crom/pyfile/relapse/mlp/mlpcsvpt"

import csv

# 결과를 저장할 새로운 CSV 파일 열기
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['pt_filepath', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader() # 헤더 쓰기

    # CSV 파일이 있는 디렉토리 경로
    

    # CSV 파일들을 읽기
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_directory, filename)
            
            try:
                # CustomDataset과 DataLoader를 생성
                custom_dataset = CustomDataset(csv_path, transform=transform)
                custom_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=len(custom_dataset))

                for data in custom_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                
                # feature 추출
                features = extract_features(model, custom_dataloader)

                # pt 파일로 저장 (파일 이름은 기존 CSV 파일과 동일하게 하되 확장자만 변경)
                pt_filepath = csv_path.replace(".csv", ".pt")
                torch.save(features, pt_filepath)

                # 라벨 추출
                label = custom_dataset.label_arr[0]  # 라벨이 하나라면 이렇게 하십시오. 아니면 더 복잡한 로직이 필요합니다.

                # pt 파일 경로와 라벨을 새로운 CSV 파일에 쓰기
                writer.writerow({'pt_filepath': pt_filepath, 'label': label})
            except Exception as e:
                print(f"Error processing file: {csv_path}")
                print(e)



a=torch.load('/home/minkyoon/crom/pyfile/relapse/mlp/mlpcsvpt/1907407226.pt')

