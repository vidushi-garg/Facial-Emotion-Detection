import torch.nn as nn
import torchvision
import sys

def define_model_densenet161(num_classes):

    model = torchvision.models.densenet161(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(2208, num_classes)

    return model
    # print(model)
    # sys.exit

# define_model_densenet161(2)