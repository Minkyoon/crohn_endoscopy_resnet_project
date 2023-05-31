import random
import os 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



root_dir='/home/minkyoon/2023_crohn/data/splitdata/'

covid_train_path = os.path.join(root_dir, 'train', '1')

covid_files      = [os.path.join(covid_train_path, x) for x in os.listdir(covid_train_path)]
covid_images    =  [np.load(x) for x in random.sample(covid_files, 5)]

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(covid_images):
    plt.subplot(int(len(covid_images) / columns+1), columns, i + 1)
    plt.imshow(image)



#  랜덤하게 5개 확인하기 






import os

def print_files_in_dir(root_dir, prefix):
    files = os.listdir(root_dir)
    lab = ['1', "0"]
    for i, file in enumerate(files):
    
        path = os.path.join(root_dir, file)
        file_list = os.listdir(path)
       
        
        print(f"{prefix} 데이터의 {lab[i]} 수: {len(file_list)}")
        
    





phase = "train"
print_files_in_dir(root_dir + f"{phase}", phase)

phase = "val"
print_files_in_dir(root_dir + f"{phase}", phase)
print()
phase = "test"
print_files_in_dir(root_dir + f"{phase}", phase)

"""### 데이터 전처리"""

# 필요한 모듈 불러오기

import numpy as np
import torch

from torchvision import datasets, transforms
import torch.utils.data as data

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


image_transforms = {
    
    'train': transforms.Compose([
        transforms.ToPILImage(),
        
        transforms.ToTensor(),
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        
        transforms.ToTensor(),
    ])
}



train_dataset_folder = root_dir + 'train'
val_dataset_folder = root_dir + 'val'
test_dataset_folder = root_dir + 'test'

root_dir

def npy_loader(path):
    sample = np.load(path)
    if len(sample.shape) == 3 and sample.shape[2] == 3:  # Check if the image has 3 channels
        sample = np.transpose(sample, (2, 0, 1))
    elif len(sample.shape) == 2:  # For grayscale images
        sample = np.expand_dims(sample, axis=0)
    else:
        raise ValueError(f'Invalid number of channels: {sample.shape}')
    return torch.from_numpy(sample)


    

train_dataset = datasets.DatasetFolder(root=train_dataset_folder,loader=npy_loader,extensions=('.npy'),transform=image_transforms['train'])
val_dataset = datasets.DatasetFolder(root=val_dataset_folder,loader=npy_loader,extensions=('.npy'),transform=image_transforms['valid'])
test_dataset = datasets.DatasetFolder(root=test_dataset_folder,loader=npy_loader,extensions=('.npy'),transform=image_transforms['valid'])



# 폴더에 여러개가 들어가 있는데?? noncovid 이랑 등등등

# DataLoader을 위한 hyperparameter 설정

train_params = {
    'batch_size': 128,
    'shuffle': True,
    'num_workers': 1,
    'drop_last': False}

    #num workers?

valid_params = {
    'batch_size': 128,
    'shuffle': False,
    'num_workers': 1,
    'drop_last': False}





train_loader = data.DataLoader(train_dataset, **train_params)
valid_loader = data.DataLoader(val_dataset, **valid_params)
test_loader = data.DataLoader(test_dataset, **valid_params)


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

## nn.Module 뜯어봐야 겠당
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
        ##super().__init__() 해도 동일할걸?

        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.fc.in_features
        self.num_ftrs = num_ftrs
        ##이게 무슨뜻이지?
        
        for name, param in model.named_parameters():
            if 'layer2' in name:
                break
            param.requires_grad = False

            ##데이터 적을땐 트레이닝 안시켜도된다?

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



"""# 모델 학습 (Model training)

### 모델 학습을 위한 설정
"""

# 학습에 필요한 변수 설정

# 학습 진행에 필요한 hyperparameter 

learning_rate = 0.001
train_epoch   = 100

# optimizer 

opt     = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

import copy
from prettytable import PrettyTable
from time import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

"""### 모델 학습 진횅"""

loss_history_train = []
loss_history_val = []

max_acc = 0

# 모델 GPU 메모리에 올리기
model = model.to(device)

# Best 모델 초기화
model_best = copy.deepcopy(model)

# 결과 정리를 위한 PrettyTable
valid_metric_record = []
valid_metric_header = ["# epoch"] 
valid_metric_header.extend(["Accuracy", "sensitivity", "specificity", "roc_score"])
table = PrettyTable(valid_metric_header)

float2str = lambda x:'%0.4f'%x # float 소숫점 4자리까지만 str로 바꾸기 

# 학습 진행
print('--- Go for Training ---')
# 학습 시작 시간 기록 
t_start = time() 

for epo in range(train_epoch):
    # Model training 
    model.train()
    
    # Mini-batch 학습 
    for i, (v_i, label) in enumerate(train_loader):
        # input data gpu에 올리기 
        v_i = v_i.float().to(device) 
        # forward-pass
        output = model(v_i) 

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))

        # 각 iteration 마다 loss 기록 
        loss_history_train.append(loss.item())

        # gradient 초기화
        opt.zero_grad()
        # back propagation
        loss.backward()
        # parameter update
        opt.step()
    
    # gradient tracking X
    with torch.set_grad_enabled(False):
        
        y_pred = []
        y_score = []
        y_label = []
        # model validation
        model.eval()

        for i, (v_i, label) in enumerate(valid_loader):
            # validation 입력 데이터 gpu에 올리기
            v_i = v_i.float().to(device)

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
    
    # metric 계산
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

    # 계산한 metric 합치기
    lst = ["epoch " + str(epo)] + list(map(float2str,[accuracy, sensitivity, specificity, roc_score]))

    # 각 epoch 마다 결과값 pretty table에 기록
    table.add_row(lst)
    valid_metric_record.append(lst)
    
    # mse 기준으로 best model 업데이트
    if accuracy > max_acc:
        # best model deepcopy 
        # model_best = copy.deepcopy(model)

        best_model_wts = copy.deepcopy(model.state_dict())
        # max MSE 업데이트 
        max_acc = accuracy

    

    # 각 epoch 마다 결과 출력 
    print('Validation at Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
						 + str(sensitivity)[:7] + ', specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])

# best_model = model.load_state_dict(best_model_wts)
model_best.load_state_dict(best_model_wts)
torch.save(model_best, '/home/minkyoon/crom/pyfile/effici.pt')

# """# 모델 테스트 (model testing)"""

# # Test dataloader 확인 
for i, (v_i, label) in enumerate(test_loader):
    print(v_i.shape)
    print(label.shape)
    break

# """### 모델 테스트 진행"""

# # 테스트 진행

# model = model_best

y_pred = []
y_label = []
y_score = []
model = torch.load('/home/minkyoon/crom/pyfile/effici.pt')
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

# """### 테스트 결과 시각화"""

# # plot the roc curve    
# fpr, tpr, _ = roc_curve(y_label, y_score)
# plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
# plt.legend(loc = 'best')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()

# import seaborn as sns

# conf_matrix = conf_matrix
# ax= plt.subplot()
# sns.heatmap(conf_matrix, annot=True, ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# # labels, title and ticks
# ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
# ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['CoViD', 'NonCoViD']); ax.yaxis.set_ticklabels(['CoViD', 'NonCoViD']);

# # gradCAM 모듈 설치
# !pip install grad-cam

# """[Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)"""

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from PIL import Image

# test_transformer = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     lambda x: x[:3],
#     transforms.Normalize(mean=[0,0,0], std=[1,1,1])
# ])

# def show_gradCAM(model, img):
#     """gradCAM을 이용하여 활성화맵(activation map)을 이미지 위에 시각화하기
#     args:
#     model (torch.nn.module): 학습된 모델 인스턴스
#     class_ind (int): 클래스 index [0 - NonCOVID, 1 - COVID]
#     img: 시각화 할 입력 이미지
#     """

#     # target_layers = [model.layer4[-1]] # 출력층 이전 마지막 레이어 가져오기
#     target_layers = [model.model_image.features[-2][-1]]
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True) 

#     inp = test_transformer(img).unsqueeze(0) # 입력 이미지 transform
#     targets = [ClassifierOutputTarget(1)] # 타겟 지정
#     grayscale_cam = cam(input_tensor=inp, targets=targets)
#     grayscale_cam = grayscale_cam[0, :]

#     # 활성화맵을 이미지 위에 표시
#     visualization = show_cam_on_image(inp.squeeze(0).permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True) 

#     pil_image=Image.fromarray(visualization)
#     return pil_image

# covid = Image.open(root_dir + 'Covid_Test/1_CT_COVID/PIIS0140673620301549_0_0.png')
# covid_cam = show_gradCAM(model, covid)
# covid_cam

# """# 수고하셨습니다."""


