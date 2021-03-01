import torch.nn as nn
import torchvision
import sys

def define_model_resnet34(num_classes):
    # Simple Identity class that let's input pass without changes
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    # Load pretrain model & modify it
    model = torchvision.models.resnet34(pretrained=False)

    # If you want to do finetuning then set requires_grad = False
    # Remove these two lines if you want to train entire model,
    # and only want to load the pretrain weights.
    for param in model.parameters():
        param.requires_grad = False

    # model.avgpool = Identity()
    model.fc = nn.Sequential(nn.Linear(512, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, num_classes))


    return model
    # print(model)
    # sys.exit

# define_model_resnet34()