import torch
import torchvision.models as models

# Load pretrained ResNet50
resnet50 = models.resnet50(pretrained=True)

# Create new model till layer3
features = list(resnet50.children())[:-3] # Remove layer4 and layers after that
features_extractor = torch.nn.Sequential(*features)


print(features)

# Now, features_extractor will give you 1024-dimensional output
