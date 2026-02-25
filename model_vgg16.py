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
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8) # 4 coordinate + 4 class
        )

    def forward(self, x):
        x = self.feature_extract(x)
        return self.fc_layers(x)



if __name__ == '__main__':
    model = Testmodel()
    print(model)
    input = torch.rand(1, 3, 448, 448) #batch：数量
    output = model(input)
    print(output)
    print(output.shape) #[1, 512, 14, 14]
