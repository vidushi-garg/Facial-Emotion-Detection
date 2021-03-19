import torch.nn as nn
import torchvision

def define_model_densenet161(num_classes):

    model = torchvision.models.densenet161(pretrained=False)

    # finetuning -> set requires_grad = False
    # Do not want to load the pretrain weights.
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(2208, num_classes)

    return model