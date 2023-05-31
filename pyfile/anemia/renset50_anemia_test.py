import random
import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 디바이스 설정 (GPU 사용 가능하면 GPU 사용하도록)
device = torch.device("cuda:1" )

random_seed = 2022

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 데이터 transform 적용하기 


# 데이터를 처리하기 위한 transform을 설정합니다.
# ToTensor를 사용해 numpy array를 Tensor로 바꿔줍니다.
transform = transforms.Compose([
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
        image = np.load(img_path,)    # npy 파일을 로드합니다.

        if self.transform:
            image = self.transform(image)

        return image, label

# 사용자 정의 Dataset 클래스를 이용하여 데이터셋을 로드합니다.
train_dataset = CustomImageDataset(csv_file='/home/minkyoon/2023_crohn/data/anemia_class_data/train.csv', transform=transform)
test_dataset = CustomImageDataset(csv_file='/home/minkyoon/2023_crohn/data/anemia_class_data/test.csv', transform=transform)
valid_dataset = CustomImageDataset(csv_file='/home/minkyoon/2023_crohn/data/anemia_class_data/valid.csv', transform=transform)

    

# DataLoader을 위한 hyperparameter 설정

train_params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 1,
    'drop_last': False}

    #num workers?

valid_params = {
    'batch_size':128,
    'shuffle': False,
    'num_workers': 1,
    'drop_last': False}



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, **train_params)
valid_loader =torch.utils.data.DataLoader(dataset=valid_dataset, **valid_params)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, **valid_params)







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
        self.dropout = nn.Dropout(0.1) # dropout 적용

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

model = Classifier(model_image, **config)
model




# 학습 진행에 필요한 hyperparameter 
learning_rate = 0.001
train_epoch   = 100
opt     = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# """# 모델 테스트 (model testing)"""

# # Test dataloader 확인 
for i, (v_i, label) in enumerate(test_loader):
    print(v_i.shape)
    print(label.shape)
    break



#  테스트 진행



y_pred = []
y_label = []
y_score = []
model = torch.load('/home/minkyoon/crom/pyfile/resnet50_anemia_epo100_real.pt')
model.eval()
for i, (v_i, label) in enumerate(test_loader):
    # input data gpu에 올리기 
    v_i = v_i.float().to(device)

    with torch.set_grad_enabled(False):
        # forward-pass
        output = model(v_i)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 

        pred = output.argmax(dim=1, keepdim=True)
        score = nn.Softmax(dim = 1)(output)[:,1]

        # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
        pred = pred.cpu().numpy()
        score = score.cpu().numpy()
        label = label.cpu().numpy()

    # 예측값, 참값 기록하기
    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + pred.flatten().tolist()
    y_score = y_score + score.flatten().tolist()

# # metric 계산
classification_metrics = classification_report(y_label, y_pred,
                    target_names = ['0', '1'],
                    output_dict= True)
# sensitivity is the recall of the positive class
sensitivity = classification_metrics['0']['recall']
# specificity is the recall of the negative class 
specificity = classification_metrics['1']['recall']
# accuracy
accuracy = classification_metrics['accuracy']
# confusion matrix
conf_matrix = confusion_matrix(y_label, y_pred)
# roc score
roc_score = roc_auc_score(y_label, y_score)

# 각 epoch 마다 결과 출력 


print('Validation, Accuracy: ' + str(accuracy)[:7] + ', Sensitivity: ' 
      + str(sensitivity)[:7] + ', Specificity: ' + str(f"{specificity}")[:7] 
      + ', ROC Score: ' + str(roc_score)[:7])

# """### 테스트 결과 시각화"""

# plot the roc curve    
fpr, tpr, _ = roc_curve(y_label, y_score)
plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
plt.legend(loc = 'best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc.png')
plt.show()

import seaborn as sns

conf_matrix = conf_matrix
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='d',ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.savefig('confusition_anemia.png')

# # gradCAM 모듈 설치
# !pip install grad-cam

y_pred = []
y_label = []
y_score = []
model = torch.load('/home/minkyoon/crom/pyfile/resnet50_anemia_epo100_real.pt')
model.eval()
for i, (v_i, label) in enumerate(test_loader):
    # input data gpu에 올리기 
    v_i = v_i.float().to(device)

    with torch.set_grad_enabled(False):
        # forward-pass
        output = model(v_i)

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 
        loss_history_val.append(loss.item())

        pred = output.argmax(dim=1, keepdim=True)
        score = nn.Softmax(dim = 1)(output)[:,1]

        # 예측값, 참값 cpu로 옮기고 numpy 형으로 변환
        pred = pred.cpu().numpy()
        score = score.cpu().numpy()
        label = label.cpu().numpy()

    # 예측값, 참값 기록하기
    y_label = y_label + label.flatten().tolist()
    y_pred = y_pred + pred.flatten().tolist()
    y_score = y_score + score.flatten().tolist()

# # metric 계산
classification_metrics = classification_report(y_label, y_pred,
                    target_names = ['0', '1'],
                    output_dict= True)
# sensitivity is the recall of the positive class
sensitivity = classification_metrics['0']['recall']
# specificity is the recall of the negative class 
specificity = classification_metrics['1']['recall']
# accuracy
accuracy = classification_metrics['accuracy']
# confusion matrix
conf_matrix = confusion_matrix(y_label, y_pred)
# roc score
roc_score = roc_auc_score(y_label, y_score)

# # 각 epoch 마다 결과 출력 
# print('Validation at Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
#                         + str(sensitivity)[:7] + ' specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])

print('Validation, Accuracy: ' + str(accuracy)[:7] + ', Sensitivity: ' 
      + str(sensitivity)[:7] + ', Specificity: ' + str(f"{specificity}")[:7] 
      + ', ROC Score: ' + str(roc_score)[:7])
"""### 테스트 결과 시각화"""

# plot the roc curve    
fpr, tpr, _ = roc_curve(y_label, y_score)
plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
plt.legend(loc = 'best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

import seaborn as sns

conf_matrix = conf_matrix
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='d', ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

test_transformer = transforms.Compose([
    
    transforms.ToTensor(),
    lambda x: x[:3],
    
])




def show_gradCAM(model, img):
    """gradCAM을 이용하여 활성화맵(activation map)을 이미지 위에 시각화하기
    args:
    model (torch.nn.module): 학습된 모델 인스턴스
    class_ind (int): 클래스 index [0 - NonCOVID, 1 - COVID]
    img: 시각화 할 입력 이미지
    """

    # target_layers = [model.layer4[-1]] # 출력층 이전 마지막 레이어 가져오기
    target_layers = [model.model_image.features[-2][-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True) 

    inp = test_transformer(img).unsqueeze(0) # 입력 이미지 transform
    targets = [ClassifierOutputTarget(1)] # 타겟 지정
    grayscale_cam = cam(input_tensor=inp, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # 활성화맵을 이미지 위에 표시
    visualization = show_cam_on_image(inp.squeeze(0).permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True) 

    pil_image=Image.fromarray(visualization)
    return pil_image

import numpy as np
from PIL import Image

# Load numpy array
crohn = np.load('/home/minkyoon/2023_crohn/data/anemia_class_data/0/1804306918b00012.npy')

# Normalize to the range 0-255
crohn = ((crohn - crohn.min()) * (1/(crohn.max() - crohn.min()) * 255)).astype('uint8')

# Convert to an image
crohn = Image.fromarray(crohn)

covid_cam = show_gradCAM(model, crohn)
covid_cam



