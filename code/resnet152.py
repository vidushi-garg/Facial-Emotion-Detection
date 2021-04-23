import torch.nn as nn
import torchvision

def define_model_resnet152(num_classes):

    # Load pretrain model & modify it
    model = torchvision.models.resnet152(pretrained=True)

    model.fc = nn.Sequential(nn.Linear(2048, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, num_classes))


    return model