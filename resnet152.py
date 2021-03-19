import torch.nn as nn
import torchvision

def define_model_resnet152(num_classes):

    # Load pretrain model & modify it
    model = torchvision.models.resnet152(pretrained=True)

    # finetuning -> set requires_grad = False
    # Do not want to load the pretrain weights.
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, num_classes))


    return model