import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
no_resBlocks = 16
HR_shape = 96
train_data_path = '../data/train'
val_data_path = '../data/val'
advLossFactor = 0.001
VGGLossFactor = 0.006
mse_lr = 0.0001
mse_epochs = 700
initial_lr = 0.0001
second_lr = 0.00001
gan_epochs = 140
batch_size = 16
images_to_eval = 10
no_workers = 8
