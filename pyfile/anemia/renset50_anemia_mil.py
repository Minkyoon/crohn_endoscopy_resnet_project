


x
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
        # forward-pass
        output = model(v_i) 

        # 미리 정의한 손실함수(MSE)로 손실(loss) 계산 
        loss = loss_fn(output, label.to(device))
        
 #       wandb.log({"Train Loss": loss.item()})

        # 각 iteration 마다 loss 기록 
        
        
        
        
  

        # gradient 초기화
        opt.zero_grad()
        # back propagation
        loss.backward()
        # parameter update
        opt.step()
        epoch_train_loss += loss.item()
        n_batches_train += 1
    loss_history_train.append(epoch_train_loss / n_batches_train)
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
            epoch_val_loss += loss.item()
            n_batches_val += 1
        loss_history_val.append(epoch_val_loss / n_batches_val)
        #wandb.log({"Validation Loss": epoch_val_loss / n_batches_val, "Train Accuracy": accuracy, "Train Sensitivity": sensitivity, "Train Specificity": specificity, "Train ROC Score": roc_score})
    
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

best_model = model.load_state_dict(best_model_wts)
model_best.load_state_dict(best_model_wts)
torch.save(model_best, 'resnet50_anemia_mil.pt')
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
save_path = 'resnet50_anemia_mil_loss.png'  # 원하는 경로와 파일명으로 변경하세요.
plot_loss_curve(loss_history_train, loss_history_val, save_path)

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



model = torch.load('resnet50_anemia_mil.pt')
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
plt.savefig('resnet50_anemia_mil_roc.png')
plt.show()
plt.close()

import seaborn as sns

conf_matrix = conf_matrix
ax= plt.subplot()
sns.heatmap(conf_matrix, annot=True, fmt='d',ax = ax, cmap = 'Blues'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
plt.savefig('resnet50_anemia_mil_confu.png')
plt.close()

result_string = ('Validation, Accuracy: ' + str(accuracy)[:7] + ', Sensitivity: ' 
      + str(sensitivity)[:7] + ', Specificity: ' + str(f"{specificity}")[:7] 
      + ', ROC Score: ' + str(roc_score)[:7])
with open('resnet50_anemia_mil.txt', 'w') as f:
    f.write(result_string)


