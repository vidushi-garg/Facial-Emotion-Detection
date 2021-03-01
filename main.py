import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from resnet34 import define_model_resnet34
from densenet161 import define_model_densenet161

from PIL import Image

#Set device
device = 'cpu'

#Hyperparameters
in_channel = 3
num_classes = 7
learning_rate = 0.001
batch_size = 2
num_epochs = 4
load_model = False

class image_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        if self.annotations.iloc[index, 1] == 'HAPPINESS':
            y_label = torch.tensor(int(0))
        elif self.annotations.iloc[index, 1] == 'DISGUST':
            y_label = torch.tensor(int(1))
        elif self.annotations.iloc[index, 1] == 'ANGER':
            y_label = torch.tensor(int(2))
        elif self.annotations.iloc[index, 1] == 'NEUTRAL':
            y_label = torch.tensor(int(3))
        elif self.annotations.iloc[index, 1] == 'SURPRISE':
            y_label = torch.tensor(int(4))
        elif self.annotations.iloc[index, 1] == 'FEAR':
            y_label = torch.tensor(int(5))
        elif self.annotations.iloc[index, 1] == 'SADNESS':
            y_label = torch.tensor(int(6))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

#Load Dataset
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0],std=[1.0])
])


path1 = 'data/IMFDB_final/IMFDB_final/AamairKhan/3Idiots/images1/'
path2 = 'data/IMFDB_final/IMFDB_final/AamairKhan/3Idiots/images3/'

listing = os.listdir(path1)
for file in listing:
    im = Image.open(path1 + file)
    imsize=(256,256)
    if imsize[0] != -1 and im.size != imsize:
        if imsize[0] > im.size[0]:
            im = im.resize(imsize, Image.BICUBIC)
        else:
            im = im.resize(imsize, Image.ANTIALIAS)
    im.save(path2 + file)

dataset = image_Dataset(csv_file = 'data/IMFDB_final/IMFDB_final/AamairKhan/3Idiots/3Idiots.csv', root_dir = 'data/IMFDB_final/IMFDB_final/AamairKhan/3Idiots/images3',
                             transform = my_transforms)


train_set, test_set = torch.utils.data.random_split(dataset, [7, 2])
# train_set, test_set = torch.utils.data.random_split(dataset, [5,3])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#Initialize Network
net = define_model_resnet34(num_classes)
# net = define_model_densenet161(num_classes)
# net.to(device=device)


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

#Saving the model after every 2 epochs
def save_checkpoint(checkpoint,filename = "checkpoint.pth.tar"):
    torch.save(checkpoint,filename)

#Load the saved model
def load_checkpoint(checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load("checkpoint.pth.tar"))

#Train the network
for i in range(num_epochs):
    epoch = i+1
    losses = []

    if epoch%2==0:
        checkpoint = {'state_dict':net.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx,(data,targets) in enumerate(train_loader):
        #Get data to cuda
        # data = data.to(device=device)
        targets = targets.to(device=device)

        #Forward
        scores = net.forward(data).to(device=device)
        loss = criterion(scores,targets)

        losses.append(loss.item())
        # torch.save('batchLoss.csv',loss)

        # x_np = loss.cpu().numpy()
        # x_df = pd.DataFrame(x_np)
        # x_df.to_csv('batchLoss.csv')

        #Backward
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent or adam step
        optimizer.step()

        print(f'Loss at batch number {batch_idx} is {loss}')

    print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            scores = model(x).to(device=device)
            # print(scores)
            _, predictions = scores.max(1)
            # print(predictions)
            num_correct += (predictions == y.to(device=device)).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, net)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, net)

