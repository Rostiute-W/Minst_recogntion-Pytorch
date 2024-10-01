import torch
from PIL import Image
from torch.utils.data import Dataset

# --------------------------------------------------------------------#
#                           数据打包
# --------------------------------------------------------------------#

class MyDataSet(Dataset):
    def __init__(self, images_path, images_class, input_shape, transforms=None):
        self.images_path = images_path
        self.images_class = images_class
        self.input_shape = input_shape
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, item):
        image_path = self.images_path[item]
        image_class = self.images_class[item]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.input_shape)
        if self.transforms:
            image = self.transforms(image)
        return image, image_class
    
    @staticmethod
    def collate_fn(batchsize):
        images, labels = tuple(zip(*batchsize))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels