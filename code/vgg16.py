import torch.nn as nn
import torchvision

def define_model_vgg16(num_classes):
    model = torchvision.models.vgg16(pretrained=True)

    model.classifier[6] = nn.Linear(4096,num_classes)


    return model