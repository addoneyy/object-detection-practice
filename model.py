import torch.onnx
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from yolo_dataset import YOLODataset


class Testmodel(nn.Module):
    def __init__(self):
        super().__init__()
        #彩色图片有3个通道
        self.seq = nn.Sequential(
            # nn.Conv2d(3, 20, 5),
            # nn.MaxPool2d(2)
            nn.AdaptiveAvgPool2d((256,256))
        )

    def forward(self, x):
        return self.seq(x)



if __name__ == '__main__':
    model = Testmodel()
    dataset = YOLODataset(r"/Users/zoe/Downloads/datatest/HelmetDataset-YOLO-Train-4425fa0029f0/images",
                          r"/Users/zoe/Downloads/datatest/HelmetDataset-YOLO-Train-4425fa0029f0/labels",
                          transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Resize((512,512))
                          ]),
                          None)
    image, target = dataset[0]
    output = model(image)
    # print(output)
    # print(model)
    #模型可视化-把模型导出为onnx格式
    # torch.onnx.export(model, image, "test-conv2d.onnx")
    print(output.shape)