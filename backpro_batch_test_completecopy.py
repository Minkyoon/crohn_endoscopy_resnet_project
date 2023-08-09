
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from utils.utils import initialize_weights
import numpy as np
import pandas as pd
from utils.svm import SmoothTop1SVM
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import copy
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns




                   
                   



parser = argparse.ArgumentParser(description='batch, epoch')
parser.add_argument('--batch', type=int, default=1, 
                    help='batch')
parser.add_argument('--epoch', type=int, default=100 ,help='epoch default=100')
args = parser.parse_args() 



            


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

## 연구에서는 그냥 cross entropy를쓰네
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
        instance_loss_fn= SmoothTop1SVM(n_classes=2), subtyping=False):
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
        p_targets = self.create_positive_targets(self.k_sample, device=device)
        n_targets = self.create_negative_targets(self.k_sample, device=device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)        
        
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        self.instance_loss_fn=self.instance_loss_fn.to(device)
        logits = logits.to(device)
        all_targets = all_targets.to(device)        
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
       # print(f'h는 이거 \n{h.shape}\n{h}')
       # print(f'A는이거 \n {A.shape}')        
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
        
    
    def forward(self, x, label=None, instance_loss_fn=None):
        # Feature Extraction
        batch_size, num_images, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = x.float().to(device)  # Convert to float
        h = self.features(x).view(batch_size, num_images, -1)
        h = h.squeeze().to(device)

        
        # CLAM
        label=label.to(device)
        ##만약 crossentropy 쓸거면 바꿔야됨
        # instance_loss_fn = nn.CrossEntropyLoss()
        instance_loss_fn = SmoothTop1SVM(n_classes = 2)
        instance_loss_fn = instance_loss_fn.to(device)
        
        logits, Y_prob, Y_hat, A_raw, results_dict = self.clam(h, label=label,)
        
        return logits, Y_prob, Y_hat, A_raw, results_dict


# 데이터 로더 생성
dataset = CustomDataset(csv_file="/home/minkyoon/crom/pyfile/relapse/mlp/mlpcsvpt/pyfile/test1.csv")


##  밑에  부분은 데이터셋 분할해서 test train valid 나누는 과정인데, 실제 데이터가지고는
## 이렇게 말고 따로 데이터셋 csv파일 만드는게 좋을듯

dataset_size = len(dataset)
print(dataset_size)
val_test_size = int(dataset_size * 0.2)  # total size for validation and test
train_size= dataset_size - val_test_size
val_size = val_test_size //2
test_size = val_test_size - val_size

train_size = dataset_size - val_test_size

train_dataset, val_dataset, test_dataset = data.random_split(dataset, [train_size, val_size, test_size])

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels
train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=False, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=custom_collate_fn)


# 통합 모델 초기화
device = torch.device("cuda:3" )
model2 = IntegratedModel(feature_extractor=model, clam_model=CLAM_SB()).to(device)
#model2= model2.to(device)

# 손실 함수 및 옵티마이저 정의
#criterion = nn.CrossEntropyLoss()
#bag_criterion = SmoothTop1SVM(n_classes = 2)
bag_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
bag_weight = 0.7
# 학습 루프
float2str = lambda x:'%0.4f'%x
num_epochs = args.epoch  #default 100
model_best = copy.deepcopy(model2)
max_acc = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    total_train_loss = 0
    total_val_loss = 0
    for i, (images, labels) in enumerate(train_dataloader):
        # Forward pass

        total_loss = 0
        
        for j in range(len(images)):
            single_image_batch, single_label_batch = images[j].unsqueeze(0).to(device), labels[j].unsqueeze(0).to(device)
            logits, Y_prob, Y_hat, A_raw, instance_dict = model2(single_image_batch, label=single_label_batch)

            loss = bag_criterion(logits.cpu(), single_label_batch.cpu())
            instance_loss = instance_dict['instance_loss']
            total_loss += bag_weight * loss + (1-bag_weight) * instance_loss
            total_train_loss += total_loss.item()
            
    #        wandb.log({'Train Loss': total_train_loss / len(train_dataloader)})
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {total_loss.item()}')
        
    train_losses.append(total_train_loss / len(train_dataloader))
    
    with torch.set_grad_enabled(False):
        y_pred = []
        y_score = []
        y_label = []
        model.eval()
        
        for i, (images, labels) in enumerate(val_dataloader):
            total_loss = 0
        
            for j in range(len(images)):
                single_image_batch, single_label_batch = images[j].unsqueeze(0).to(device), labels[j].unsqueeze(0).to(device)
                logits, Y_prob, Y_hat, A_raw, instance_dict = model2(single_image_batch, label=single_label_batch)

                loss = bag_criterion(logits.cpu(), single_label_batch.cpu())
                instance_loss = instance_dict['instance_loss']
                total_loss += bag_weight * loss + (1-bag_weight) * instance_loss
                total_val_loss += total_loss.item()

                # Move the tensors to cpu for numpy conversion
                pred = logits.argmax(dim=1, keepdim=True).cpu().numpy()
                score = nn.Softmax(dim = 1)(logits)[:,1].cpu().numpy()
                label = single_label_batch.cpu().numpy()

                y_label += label.flatten().tolist()
                y_pred += pred.flatten().tolist()
                y_score += score.flatten().tolist()
                
        # wandb.log({
        #     'Validation Loss': total_val_loss / len(val_dataloader),
        #     'Accuracy': accuracy,
        #     'Sensitivity': sensitivity,
        #     'Specificity': specificity,
        #     'ROC AUC': roc_score
        # })
                
        val_losses.append(total_val_loss / len(val_dataloader))

    print(y_label)
    print(y_pred)        
    classification_metrics = classification_report(y_label, y_pred,
                        target_names = ['0', '1'],
                        output_dict= True)        
    sensitivity = classification_metrics['0']['recall']
    # specificity is the recall of the negative class 
    specificity = classification_metrics['1']['recall']
    # accuracy
    accuracy = classification_metrics['accuracy']
    # confusion matrix
    conf_matrix = confusion_matrix(y_label, y_pred)
    # roc score
    roc_score = roc_auc_score(y_label, y_score)
    lst = ["epoch " + str(epoch)] + list(map(float2str,[accuracy, sensitivity, specificity, roc_score]))
    if accuracy > max_acc:

        best_model_wts = copy.deepcopy(model2.state_dict())
        max_acc = accuracy 
        
        print('Validation at Epoch '+ str(epoch + 1) + ' , Accuracy: ' + str(accuracy)[:7] + ' , sensitivity: '\
						 + str(sensitivity)[:7] + ', specificity: ' + str(f"{specificity}") +' , roc_score: '+str(roc_score)[:7])       
    

model2.load_state_dict(best_model_wts)

model_best.load_state_dict(best_model_wts)
torch.save(model_best.state_dict(), 'resnet50_epoch100_trans2.pt')

print('Finished Training')


def plot_loss_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10,7))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


save_path = 'loss_curve.png'
plot_loss_curve(train_losses, val_losses, save_path)



model2 = IntegratedModel(feature_extractor=model, clam_model=CLAM_SB()).to(device)
model2.load_state_dict(torch.load('resnet50_epoch100_trans2.pt'))

model2.eval()  # set the model to evaluation mode

y_pred_test = []
y_score_test = []
y_label_test = []

with torch.no_grad():  # gradients are not needed for testing
    for images, labels in test_dataloader:
        total_loss_test = 0
        
        for j in range(len(images)):
            single_image_batch, single_label_batch = images[j].unsqueeze(0).to(device), labels[j].unsqueeze(0).to(device)
            logits, Y_prob, Y_hat, A_raw, instance_dict = model2(single_image_batch, label=single_label_batch)

            loss = bag_criterion(logits.cpu(), single_label_batch.cpu())
            instance_loss = instance_dict['instance_loss']
            total_loss_test += bag_weight * loss + (1-bag_weight) * instance_loss

            pred = logits.argmax(dim=1, keepdim=True).cpu().numpy()
            score = nn.Softmax(dim = 1)(logits)[:,1].cpu().numpy()
            label = single_label_batch.cpu().numpy()

            y_label_test += label.flatten().tolist()
            y_pred_test += pred.flatten().tolist()
            y_score_test += score.flatten().tolist()

classification_metrics_test = classification_report(y_label_test, y_pred_test,
                    target_names = ['0', '1'],
                    output_dict= True) 

sensitivity_test = classification_metrics_test['0']['recall']
specificity_test = classification_metrics_test['1']['recall']
accuracy_test = classification_metrics_test['accuracy']
conf_matrix_test = confusion_matrix(y_label_test, y_pred_test)
roc_score_test = roc_auc_score(y_label_test, y_score_test)

print('Testing results: ')
print('Accuracy: ', accuracy_test)
print('Sensitivity: ', sensitivity_test)
print('Specificity: ', specificity_test)
print('ROC Score: ', roc_score_test)
print('Confusion matrix: \n', conf_matrix_test)





# ROC curve
fpr_test, tpr_test, _ = roc_curve(y_label_test, y_score_test)
plt.figure(figsize=(8,8))
plt.plot(fpr_test, tpr_test, label = "Area under ROC = {:.4f}".format(roc_score_test))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.savefig('roc_test.png')
plt.show()
plt.close()

# Confusion matrix
plt.figure(figsize=(8,8))
conf_matrix_test = conf_matrix_test
ax= plt.subplot()
sns.heatmap(conf_matrix_test, annot=True, fmt='d',ax = ax, cmap = 'Blues')  # annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['0', '1'])
ax.yaxis.set_ticklabels(['0', '1'])
plt.savefig('confusion_test.png')
plt.close()

# Save results in a text file
result_string_test = ('Test, Accuracy: ' + str(accuracy_test)[:7] 
                     + ', Sensitivity: ' + str(sensitivity_test)[:7] 
                     + ', Specificity: ' + str(specificity_test)[:7] 
                     + ', ROC Score: ' + str(roc_score_test)[:7])
with open('results_test.txt', 'w') as f:
    f.write(result_string_test)
