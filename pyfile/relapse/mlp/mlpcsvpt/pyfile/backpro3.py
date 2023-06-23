
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from utils.utils import initialize_weights
import numpy as np
import pandas as pd
from topk.svm import SmoothTop1SVM
import matplotlib.pyplot as plt
from PIL import Image



            


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
       # print(f'attention에서 A는이거 \n {A.shape}')   
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
        device=torch.device("cuda:1")
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
        p_targets = self.create_positive_targets(self.k_sample, device=device)
        n_targets = self.create_negative_targets(self.k_sample, device=device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)        
        
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        self.instance_loss_fn=self.instance_loss_fn.to(device)
        logits = logits.cpu()
        all_targets = all_targets.cpu()        
        instance_loss = self.instance_loss_fn(logits, all_targets)
        print('이거 까지 출력되나?')
        return instance_loss, all_preds, all_targets, logits
    
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

    def forward(self, h, label=None, instance_eval=True, return_features=False, attention_only=False):
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
            all_logits=[]
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    print("요거 요거 요거 되?")
                    instance_loss, preds, targets, logits = self.inst_eval(A, h, classifier)
                    print("요거 요거 안되지?")
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_logits.extend(logits.cpu().numpyt)
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
       # print(f'h는 이거 \n{h.shape}\n{h}')
       # print(f'A는이거 \n {A.shape}')        
        M = torch.mm(A, h)  
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds), 'all_logit': np.array(all_logits)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict





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



device = torch.device("cuda:1" )

model = Classifier(model_image, **config)


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






transform = transforms.Compose([
    transforms.ToTensor()])





# 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data['accession_number'].unique())

    def __getitem__(self, idx):
        accession_number = self.data['accession_number'].unique()[idx]
        print(accession_number)
        images = [self.transform(np.load(row['filepath'])) for _, row in self.data[self.data['accession_number'] == accession_number].iterrows()]
        label = self.data[self.data['accession_number'] == accession_number].iloc[0]['label']
        return torch.stack(images), label


# 통합 모델 (특징 추출기 + CLAM)
class IntegratedModel(nn.Module):
    def __init__(self, feature_extractor, clam_model):
        super(IntegratedModel, self).__init__()
        
        # Feature Extractor (ResNet50)
        self.features=feature_extractor
        self.clam=clam_model
        
    
    def forward(self, x_list, label=None):
        # Feature Extraction
        batch_outputs=[]
        for x in x_list:
            print(len(x_list))
            print(x.shape)
            batch_size, num_images, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            x = x.float().to(device)  # Convert to float
            h = self.features(x).view(batch_size, num_images, -1)
            h = h.squeeze().to(device)
            #  print(h)
            # print(h.shape)
            
            # CLAM
            label=label.to(device)
            logits, Y_prob, Y_hat, A_raw, results_dict = self.clam(h, label=label)
            batch_outputs.append((logits, Y_prob, Y_hat, A_raw, results_dict))
        
                       
        return batch_outputs


# 데이터 로더 생성
dataset = CustomDataset(csv_file="/home/minkyoon/crom/pyfile/relapse/mlp/mlpcsvpt/pyfile/test1.csv")

batch=2
if batch >= 2 :
    def custom_collate_fn(batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])        
        print(len(images))
        print(images[1].shape)
        return images, labels
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=custom_collate_fn)
    
   # data_loader = DataLoader(dataset, batch_size=batch, shuffle=False)

# 통합 모델 초기화
device = torch.device("cuda:1" )
model2 = IntegratedModel(feature_extractor=model, clam_model=CLAM_SB()).to(device)
#model2= model2.to(device)

# 손실 함수 및 옵티마이저 정의
#criterion = nn.CrossEntropyLoss()
bag_criterion = SmoothTop1SVM(n_classes = 2)
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)
train_loss = 0.
train_error = 0.
train_inst_loss = 0.
inst_count = 0
bag_weight = 0.7
# 학습 루프
num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        # Forward pass
        logits_list = []
        Y_prob_list = []
        Y_hat_list = []
        A_raw_list = []
        instance_dict_list = []
        total_loss = 0
        
        for j in range(len(images)):
            single_image_batch, single_label_batch = images[j].to(device), labels[j].unsqueeze(0).to(device)
            logits, Y_prob, Y_hat, A_raw, instance_dict = model2(single_image_batch, label=single_label_batch)
            logits_list.append(logits.cpu())
            Y_prob_list.append(Y_prob)
            Y_hat_list.append(Y_hat)
            A_raw_list.append(A_raw)
            instance_dict_list.append(instance_dict)
            
            loss = bag_criterion(logits.cpu(), single_label_batch.cpu())
            instance_loss = instance_dict['instance_loss']
            total_loss += bag_weight * loss + (1-bag_weight) * instance_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {total_loss.item()}')

print('Finished Training')