import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torchvision.models import resnet18 as tv_resnet18 
import models
import seaborn as sns

# Define the 50 specific classes
fixed_classes = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    #40, 41, 42, 43, 44, 45, 46, 47, 48, 49
]

# Define colors for the fixed classes
#colors = plt.cm.get_cmap('tab20', 50)  # Use tab20 colormap with 50 unique colors
#palette = sns.color_palette('hsv', 50)
#palette = sns.color_palette('tab20', 50)
palette1 = sns.color_palette('tab20', 20)
palette2 = sns.color_palette('tab20b', 20)
#palette3 = sns.color_palette('tab20c', 10)
all_colors = palette1 + palette2 #+ palette3

'''
colors = plt.cm.get_cmap('tab20', 50)
if len(fixed_classes) > 20:
    extra_colors = plt.cm.get_cmap('hsv', len(fixed_classes) - 20)
    all_colors = [colors(i) for i in range(20)] + [extra_colors(i) for i in range(len(fixed_classes) - 20)]
else:
    all_colors = [colors(i) for i in range(len(fixed_classes))]
'''
# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load CIFAR-100 dataset
testset = datasets.CIFAR100(root='/home/shuwen/data/', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

# Load the model
#model = getattr(models, 'CIFAR_ResNet18_byot')
model = getattr(models, 'CIFAR_ResNet18')

model = model(num_classes=100)

checkpoint_path = "path/to/checkpoint.pth.tar"


checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['net'])
model = model.eval().cuda()

# Extract features and labels
features = []
labels = []

with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        features.append(outputs.cpu().numpy())
        labels.append(targets.numpy())

features = np.concatenate(features)
labels = np.concatenate(labels)

# Filter features and labels for the fixed classes
filtered_features = []
filtered_labels = []

for class_idx in fixed_classes:
    indices = labels == class_idx
    filtered_features.append(features[indices])
    filtered_labels.append(labels[indices])

filtered_features = np.concatenate(filtered_features)
filtered_labels = np.concatenate(filtered_labels)

# Apply t-SNE to the filtered features
tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(filtered_features)

plt.figure(figsize=(10, 10))
for i, class_idx in enumerate(fixed_classes):
    indices = filtered_labels == class_idx
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], 
                label=testset.classes[class_idx], color=all_colors[i], s=10)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.savefig('cskd1111.pdf', bbox_inches='tight')
plt.show()