import torch



device=(torch.device('cuda:1'))

model = torch.load('/home/minkyoon/crom/pyfile/resnet50_anemia_epo100_real.pt')
model.eval()

from collections import defaultdict, Counter

# Create a dictionary to hold the list of predictions for each Accession Number.
predictions_by_accession = defaultdict(list)

# Assuming you have a prediction and its corresponding Accession Number,
# append the prediction to the corresponding list.
for i, (images, labels, accession_numbers) in enumerate(your_data_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    for pred, acc_num in zip(predicted, accession_numbers):
        predictions_by_accession[acc_num].append(pred.item())

# Now, calculate the most common prediction for each Accession Number.
final_predictions = {}
for acc_num, predictions in predictions_by_accession.items():
    final_predictions[acc_num] = Counter(predictions).most_common(1)[0][0]



from torch.utils.data import Dataset
from PIL import Image


transform = transforms.Compose([transforms.ToTensor()])        
    
    
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
