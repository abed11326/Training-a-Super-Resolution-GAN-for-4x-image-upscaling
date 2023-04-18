from imageData import ImageData
from torch.utils.data import DataLoader
from hypParam import *
import matplotlib.pyplot as plt
from datetime import datetime
import os

dataset = ImageData(val_data_path, training =False)
loader = iter(DataLoader(dataset, 1, shuffle=True, pin_memory=True))

def tensorToImage(tens):
    return tens.cpu().data.numpy().transpose((1,2,0))

def save(image, title, path_to_save):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title[2:], size=30)
    plt.savefig(f"{path_to_save}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}{title}.png", bbox_inches='tight', pad_inches=0.0)
    plt.close()
    # plt.savefig('hr ' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.png')

def evaluate(model, no_images, path_to_save, model_name):
    """
    This function performs visual evaluation on a model. Given the model, pass a set of low resolution images and save the resulted SR images.
    """
    model.eval()
    if not os.path.isdir(path_to_save):
        os.system(f"mkdir -p {path_to_save}")
    for i in range(no_images):
        low_res, high_res = next(loader)
        low_res = low_res.to(device)
        sup_res = model(low_res).detach()
        save(tensorToImage(low_res[0]), f'{i+1} Low Resolution', path_to_save)
        save(tensorToImage((sup_res[0] + 1) / 2), f'{i+1} {model_name}', path_to_save)
        save(tensorToImage((high_res[0] + 1) / 2), f'{i+1} Original', path_to_save)
