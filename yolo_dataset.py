import os
import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        # 获取图片文件夹下所有文件的名称，以列表的方式返回
        self.img_names = os.listdir(image_folder)
        # self.classes_list = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        #图片路径
        img_path = os.path.join(self.image_folder,img_name)
        #转换成RGB格式，打开图片
        image = Image.open(img_path).convert("RGB")
        #打开图片对应的labels文件
        #new1.png -> new1.txt
        label_name = img_name.split(".")[0] + ".txt"
        #图片对应labels文件路径
        label_path = os.path.join(self.label_folder,label_name)
        #打开labels文件
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        object_infos = label_content.strip().split("\n")
        target = []
        for object_info in object_infos:
            info_list = object_info.strip().split(" ")
            class_id = float(info_list[0])
            center_x = float(info_list[1])
            center_y = float(info_list[2])
            width = float(info_list[3])
            height = float(info_list[4])
            target.extend([class_id, center_x, center_y, width, height])
        if self.transform is not None:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    #torchvision的transforms可以实现图片从PIL格式转换为tensor格式
    #transforms.Compose([transforms.ToTensor(),xxx])
    train_dataset = YOLODataset(r"/Users/zoe/Downloads/datatest/HelmetDataset-YOLO-Train-4425fa0029f0/images", r"/Users/zoe/Downloads/datatest/HelmetDataset-YOLO-Train-4425fa0029f0/labels",
                                transforms.Compose([transforms.ToTensor()]), None)
    print(len(train_dataset))
    print(train_dataset[11])
