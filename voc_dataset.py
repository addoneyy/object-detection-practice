import os
import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCDataset(Dataset):
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        # 获取图片文件夹下所有文件的名称，以列表的方式返回
        self.img_names = os.listdir(image_folder)
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        #图片路径
        img_path = os.path.join(self.image_folder,img_name)
        #转换成RGB格式，打开图片
        image = Image.open(img_path).convert("RGB")
        #打开图片对应的labels文件
        label_name = img_name.split(".")[0] + ".xml"
        #图片对应labels文件路径
        label_path = os.path.join(self.label_folder,label_name)
        #打开labels文件
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        #将xml格式文件转换为python字典格式
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        target = []
        for obj in objects:
            object_name = obj["name"]
            #返回object_name在classes.txt列表索引
            object_class_id = self.classes_list.index(object_name)
            object_xmin = float(obj["bndbox"]["xmin"])
            object_ymin = float(obj["bndbox"]["ymin"])
            object_xmax = float(obj["bndbox"]["xmax"])
            object_ymax = float(obj["bndbox"]["ymax"])
            target.extend([object_class_id, object_xmin, object_ymin, object_xmax, object_ymax])
        #返回的数据必须是tensor类型，可以做一些特殊的操作，在深度学习中使用
        target = torch.tensor(target)
        #把image从PIL格式转换为tensor格式
        if self.transform is not None:
            image = self.transform(image)
        return image, target


if __name__ == '__main__':
    #torchvision的transforms可以实现图片从PIL格式转换为tensor格式
    #transforms.Compose([transforms.ToTensor(),xxx])
    train_dataset = VOCDataset(r"/Users/zoe/Downloads/HelmetDataset-VOC/train/images",r"/Users/zoe/Downloads/HelmetDataset-VOC/train/labels",
                               transforms.Compose([transforms.ToTensor()]),None)
    print(len(train_dataset))
    print(train_dataset[1])
