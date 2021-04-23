import torch.nn as nn
import torchvision

def define_model_densenet201(num_classes):

    model = torchvision.models.densenet201(pretrained=False)

    model.classifier = nn.Linear(1920, num_classes)

    return model
