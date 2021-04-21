import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from resnet152 import define_model_resnet152
from densenet201 import define_model_densenet201
from vgg16 import define_model_vgg16
import numpy as np
import PIL
from PIL import Image

from PIL import Image

#Set device

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
dtype1 = torch.cuda.LongTensor

# torch.cuda.set_device(1)
#Hyperparameters
in_channel = 3
num_classes = 7
learning_rate = 0.001
batch_size = 64
num_epochs = 30
load_model = False

# Width and Height of the image as given in the dataset
(width, height) = 48,48

# Convert each row of 'pixels' column into numpy array of images
def readPixels(pixelRow):
    # Split the string of pixels separated by spaces
    pixelRow = [int(pixel) for pixel in pixelRow.split(' ')]
    # Convert this list of pixel values into numpy array
    return np.asarray(pixelRow).reshape(width, height)

# class image_Dataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 4])
#         image = io.imread(img_path)
#
#         a = self.annotations.iloc[index, 1]
#         y_label = torch.tensor(int(a)).type(dtype)
#
#         if self.transform:
#             image = self.transform(image)
#             image = image.repeat(3, 1, 1)
#
#         return (image, y_label)

class image_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        pixels = self.annotations.iloc[index, 2]
        # Convert each row of 'pixels' column into numpy array of images
        image = readPixels(pixels)
        image = Image.fromarray(np.uint8(image))


        a = self.annotations.iloc[index, 1]
        y_label = torch.tensor(int(a)).type(dtype)

        if self.transform:
            image = self.transform(image)
            image = image.repeat(3, 1, 1)

        return (image, y_label)

#Load Dataset
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.507395],std=[0.2551289])
    transforms.Normalize(mean=[0.55199619],std=[0.2486985])
])



# dataset = image_Dataset(csv_file = '../dataset/emotion_dataset.csv', root_dir = '../dataset/images', transform = my_transforms)

dataset = image_Dataset(csv_file = '../dataset/emotion_dataset.csv', transform = my_transforms)

#Dividing the dataset into two parts
train_set, test_set = torch.utils.data.random_split(dataset, [25000, 10887])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#Initialize Network
# net = define_model_resnet152(num_classes)
# net = define_model_densenet201(num_classes)
net = define_model_vgg16(num_classes)
net = net.type(dtype)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

#Saving the model after every epoch
def save_checkpoint(checkpoint,filename = "checkpoint.pth.tar"):
    torch.save(checkpoint,filename)

#Load the saved model
def load_checkpoint(checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if load_model:
    load_checkpoint(torch.load("checkpoint.pth.tar"))

counter = False
previous_loss = 0;
#Train the network
for i in range(num_epochs):
    epoch = i+1
    losses = []

    if epoch:
        checkpoint = {'state_dict':net.state_dict(),'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx,(data,targets) in enumerate(train_loader):
        #Get data to cuda
        data = data.type(dtype)
        targets = targets.type(dtype1)

        #Forward
        scores = net.forward(data)
        loss = criterion(scores,targets).type(dtype)


        losses.append(loss.item())

        #Backward
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent or adam step
        optimizer.step()

        print(f'Loss at batch number {batch_idx} is {loss}')

    current_loss = sum(losses) / len(losses)
    #Manually updating the learning rate
    if counter:
        if (current_loss > previous_loss):
            lr = learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    print("Learning Rate ",learning_rate)
    counter = True
    previous_loss = current_loss;
    print(f'Cost at epoch {epoch} is {current_loss}')


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            scores = model(x.type(dtype))
            _, predictions = scores.max(1)
            num_correct += (predictions == y.type(dtype)).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, net)

net.eval()
print("Checking accuracy on Test Set")
check_accuracy(test_loader, net)
net.train()
