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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 디바이스 설정 (GPU 사용 가능하면 GPU 사용하도록)
device = torch.device("cuda:3" )

random_seed = 2022

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 데이터 transform 적용하기 



transform = transforms.Compose([
      
      
    transforms.ToTensor(), 
    transforms.RandomHorizontalFlip(),   
    transforms.RandomVerticalFlip(),     
    transforms.RandomRotation(30), 
    
])

transform_valid = transforms.Compose([
    transforms.ToTensor(), 
           
    
])

# 사용자 정의 Dataset 클래스
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = np.load(img_path).astype(np.float32),    # npy 파일을 로드합니다.

        if isinstance(image, tuple):
            image = image[0]
        if np.isnan(image).any():
            raise ValueError(f"NaN value found in input image at index {idx}")
            
        
        
        
        if self.transform:
           
            
            image = self.transform(image)
        
        # 레이블 데이터를 torch.Tensor로 변환하면서 데이터 타입을 float32로 변경합니다.
        label = torch.tensor(label).float()    

        return image, label

# 사용자 정의 Dataset 클래스를 이용하여 데이터셋을 로드합니다.
train_dataset = CustomImageDataset(csv_file='/home/minkyoon/crom/pyfile/crp_regression/train.csv', transform=transform)
test_dataset = CustomImageDataset(csv_file='/home/minkyoon/crom/pyfile/crp_regression/test.csv', transform=transform_valid)
valid_dataset = CustomImageDataset(csv_file='/home/minkyoon/crom/pyfile/crp_regression/valid.csv', transform=transform_valid)

    

# DataLoader을 위한 hyperparameter 설정

train_params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 1,
    'drop_last': False}

    #num workers?

valid_params = {
    'batch_size': 64,
    'shuffle': False,
    'num_workers': 1,
    'drop_last': False}



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, **train_params)
valid_loader =torch.utils.data.DataLoader(dataset=valid_dataset, **valid_params)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, **valid_params)





# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
 #                                             num_workers=4)



# Train DataLoader 데이터 확인해보기

for x, y in train_loader:
    print(x.shape)
    print(y.shape)
    break

# Valid DataLoader 데이터 확인해보기

for x, y in valid_loader:
    print(x.shape)
    print(y.shape)
    break

"""# 모델 만들기"""

from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 모델 설정 값

config = {
    # Classfier 설정
    "cls_hidden_dims" : [1024, 512, 256]
    }


class ResNet(nn.Module):
    """pretrain 된 ResNet을 이
    """
    
    def __init__(self):
        """
		Args:
			base_model : resnet18 / resnet50
			config: 모델 설정 값
		"""
        super(ResNet, self).__init__()
       
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





model_image = ResNet()
model_image

class Regression(nn.Module):
    def __init__(self, model_image, **config):
        super(Regression, self).__init__()

        self.model_image = model_image
        self.input_dim = model_image.num_ftrs
        self.dropout = nn.Dropout(0.5)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim] + self.hidden_dims + [1] # 출력 노드를 1개로 설정

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

    def forward(self, v):
        v_i = self.model_image(v)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer, do not apply activation function
                v_i = l(v_i)
            else:
                # If Not last layer, apply dropout and ReLU
                v_i = F.relu(self.dropout(l(v_i)))
        return v_i.squeeze()

model = Regression(model_image, **config)







"""# 모델 학습 (Model training)

### 모델 학습을 위한 설정
"""
import copy
from prettytable import PrettyTable
from time import time


# 학습 진행에 필요한 hyperparameter 

learning_rate = 0.00001
train_epoch   = 100

# optimizer 

# optimizer
opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss function
class MAPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return loss


loss_fn = MAPELoss()
loss_fn = torch.nn.MSELoss()



loss_history_train = []
loss_history_val = []

min_loss = float('inf')

# 모델 GPU 메모리에 올리기
model = model.to(device)

# Best 모델 초기화
model_best = copy.deepcopy(model)

# 결과 정리를 위한 PrettyTable
valid_metric_record = []
valid_metric_header = ["# epoch"] 
valid_metric_header.extend(["MSE", "MAE", "R2 Score"])
table = PrettyTable(valid_metric_header)

float2str = lambda x:'%0.4f'%x # float 소숫점 4자리까지만 str로 바꾸기 

for epo in range(train_epoch):
    # Model training 
    model.train()
    
    epoch_train_loss=0
    n_batches_train = 0
    epoch_val_loss = 0
    n_batches_val = 0
    
    # Mini-batch 학습 
    for i, (v_i, label) in enumerate(train_loader):
        # input data gpu에 올리기 
        v_i = v_i.float().to(device)
        label = label.float().to(device) 
        # forward-pass
        
        if torch.isnan(label).any() :
            print('NaN train value in input data')
        output = model(v_i) 
        output = output.view(-1)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label)
        
        # Check if loss is NaN
        if torch.isnan(loss):
            print('NaN train value in loss')

        # 각 iteration 마다 loss 기록 
        epoch_train_loss += loss.item()
        n_batches_train += 1
        
        # gradient 초기화
        opt.zero_grad()
        # back propagation
        loss.backward()
        # parameter update
        opt.step()
    loss_history_train.append(epoch_train_loss / n_batches_train)

    # gradient tracking X
    with torch.set_grad_enabled(False):
        
        y_pred = []
        y_label = []
        # model validation
        model.eval()

        for i, (v_i, label) in enumerate(valid_loader):
            # validation 입력 데이터 gpu에 올리기
            v_i = v_i.float().to(device)
            label = label.float().to(device)
            
            # Check if v_i or label contains NaNs
            if torch.isnan(v_i).any() or torch.isnan(label).any():
                print('NaN value in input data')

            # forward-pass
            output = model(v_i)
            
            # Check if the network parameters contain NaNs
            for param in model.parameters():
                if torch.isnan(param).any():
                    print('NaN value in network parameters')


            # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
            loss = loss_fn(output, label)
            # Check if loss is NaN
            if torch.isnan(loss):
                print('NaN value in loss')

            # 각 iteration 마다 loss 기록 
            epoch_val_loss += loss.item()
            n_batches_val += 1

            # 예측값, 참값 cpu로 올리고 numpy 형으로 변환
            output = output.detach().cpu().numpy()
            label = label.cpu().numpy()

            # 예측값, 참값 기록하기
            y_label = y_label + label.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
        
        
        loss_history_val.append(epoch_val_loss / n_batches_val)
    
    # 회귀 성능 지표 계산
    print(np.isnan(y_label).any())
    print(np.isnan(y_pred).any())
    mse = mean_squared_error(y_label, y_pred)
    mae = mean_absolute_error(y_label, y_pred)
    r2 = r2_score(y_label, y_pred)

    # 계산한 metric 합치기
    lst = ["epoch " + str(epo)] + list(map(float2str,[mse, mae, r2]))

    # 각 epoch 마다 결과값 pretty table에 기록
    table.add_row(lst)
    valid_metric_record.append(lst)
    
    # mse 기준으로 best model 업데이트
    if mse < min_loss:
        # best model deepcopy 
        best_model_wts = copy.deepcopy(model.state_dict())
        # min MSE 업데이트 
        min_loss = mse

    # 각 epoch 마다 결과 출력 
    print('Validation at Epoch '+ str(epo + 1) + ' , MSE: ' + str(mse)[:7] + ' , MAE: '\
						 + str(mae)[:7] + ', R2 Score: ' + str(r2)[:7])

model_best.load_state_dict(best_model_wts)
torch.save(model_best, 'resnet_regression2.pt')

import matplotlib.pyplot as plt

# 학습 곡선 그리기
def plot_loss_curve(loss_history_train, loss_history_val, save_path):
    plt.figure(figsize=(10,7))
    plt.plot(loss_history_train, label='Train')
    plt.plot(loss_history_val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 이미지 저장
    plt.close()
    
    
# ... training code ...

# 학습이 끝난 후에 학습 곡선을 그립니다.
# 이미지 저장 경로 설정
save_path = 'loss_curve2.png'  # 원하는 경로와 파일명으로 변경하세요.
plot_loss_curve(loss_history_train, loss_history_val, save_path)

# """# 모델 테스트 (model testing)"""

# # Test dataloader 확인 
for i, (v_i, label) in enumerate(test_loader):
    print(v_i.shape)
    print(label.shape)
    break

# """### 모델 테스트 진행"""

# # 테스트 진행

model = torch.load('resnet_regression2.pt')
model.eval()

y_pred = []
y_label = []

for i, (v_i, label) in enumerate(test_loader):
    # input data gpu에 올리기 
    v_i = v_i.float().to(device)
    label = label.float().to(device)

    with torch.set_grad_enabled(False):
        # forward-pass
        output = model(v_i)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label)

        # 각 iteration 마다 loss 기록 

        # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
        output = output.detach().cpu().numpy()
        label = label.cpu().numpy()

    # 예측값, 참값 기록하기
    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + output.flatten().tolist()

# 회귀 성능 지표 계산
mse = mean_squared_error(y_label, y_pred)
mae = mean_absolute_error(y_label, y_pred)
r2 = r2_score(y_label, y_pred)

print('Validation, MSE: ' + str(mse)[:7] + ', MAE: ' 
      + str(mae)[:7] + ', R2 Score: ' + str(r2)[:7])

# 결과 저장
result_string = ('Validation, MSE: ' + str(mse)[:7] + ', MAE: ' 
      + str(mae)[:7] + ', R2 Score: ' + str(r2)[:7])
with open('results.txt', 'w') as f:
    f.write(result_string)


