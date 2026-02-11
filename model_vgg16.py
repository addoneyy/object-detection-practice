import torch.onnx
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg16

from yolo_dataset import YOLODataset


class Testmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract = vgg16().features

    def forward(self, x):
        return self.feature_extract(x)



if __name__ == '__main__':
    model = Testmodel()
    print(model)