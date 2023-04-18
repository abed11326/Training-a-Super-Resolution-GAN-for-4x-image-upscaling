import os
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, Resize, InterpolationMode, RandomHorizontalFlip
from torchvision.transforms.functional import rotate
from torchvision.io import read_image
import numpy as np

class ImageData(Dataset):
    def __init__(self, data_path, HR_shape=None, training=True):
        super(ImageData, self).__init__()
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.training = training
        if training:
            LR_shape = HR_shape // 4
            self.crop = RandomCrop((HR_shape, HR_shape), pad_if_needed=True)
            self.resize = Resize((LR_shape, LR_shape), InterpolationMode.BICUBIC)
            self.rand_flip = RandomHorizontalFlip()
        else:
            self.crop = RandomCrop((400, 400), pad_if_needed=True)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        file_name = self.data[index]
        image = read_image(os.path.join(self.data_path, file_name))
        image = self.crop(image)
        if self.training:
            image = self.rand_flip(image)
            if np.random.rand() < 0.5:
                image = rotate(image, 90)
            LR_image = self.resize(image) / 255.0
        else:
            LR_image = Resize((image.shape[1] // 4, image.shape[2] // 4), InterpolationMode.BICUBIC)(image) / 255.0
        HR_image = 2 * (image / 255.0) - 1
        return LR_image, HR_image
