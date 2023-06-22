import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from topk.svm import SmoothTop1SVM
            


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
    
"""


class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
        instance_loss_fn= SmoothTop1SVM(n_classes = 2), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda:3")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict



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
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


config = {
    # Classfier 설정
    "cls_hidden_dims" : [1024, 512, 256]
    }



class ResNet(nn.Module):
    """pretrain 된 ResNet을 이
    """
    
    def __init__(self):
      
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



device = torch.device("cuda:3" )

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




from torch.utils.data import Dataset
import numpy as np
import pandas as pd


transform = transforms.Compose([
    transforms.ToTensor(), 
           
    
])





    
    


class IntegratedModel(nn.Module):
    def __init__(self, feature_extractor, clam_model):
        super(IntegratedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.clam_model = clam_model

    def forward(self, x, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = self.feature_extractor(x)
        return self.clam_model(h, label, instance_eval, return_features, attention_only)
    
    
    
    
    
    




import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data['accession_number'].unique())

    def __getitem__(self, idx):
        accession_number = self.data['accession_number'].unique()[idx]
        images = [self.transform(np.load(row['filepath'])) for _, row in self.data[self.data['accession_number'] == accession_number].iterrows()]
        label = self.data[self.data['accession_number'] == accession_number].iloc[0]['label']
        return torch.stack(images), label


# 통합 모델 (특징 추출기 + CLAM)
class IntegratedModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(IntegratedModel, self).__init__()
        
        # Feature Extractor (ResNet50)
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # CLAM
        self.attention_net = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        # Feature Extraction
        batch_size, num_images, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        h = self.features(x).view(batch_size, num_images, -1)
        
        # CLAM
        A = self.attention_net(h)
        A = torch.softmax(A, dim=1)
        M = torch.bmm(A.transpose(1, 2), h)
        output = self.classifier(M.squeeze(1))
        
        return output


# 데이터 로더 생성
dataset = CustomDataset(csv_file="/home/minkyoon/crom/pyfile/relapse/mlp/mlpcsvpt/1504124957.csv")
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 통합 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntegratedModel(feature_dim=1024, num_classes=2).to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')

print('Finished Training')