import torch
from torch.nn import Module, MSELoss, BCEWithLogitsLoss
from torchvision.models import vgg19
from hypParam import *

mse = MSELoss()
bce = BCEWithLogitsLoss()

class GeneratorLoss(Module):
    def __init__(self, advLossFactor, VGGLossFactor):
        super(GeneratorLoss, self).__init__()
        self.advLossFactor = advLossFactor
        self.VGGLossFactor = VGGLossFactor
        self.init_VGG()

    def init_VGG(self):
        self.vgg = vgg19()
        self.vgg.load_state_dict(torch.load('./parameters/vgg19.pth'))
        self.vgg = self.vgg.features[:36].eval().to(device)

        for p in self.vgg.parameters():
            p.requires_grad = False
            
    def content_loss(self, fakeImages, high_res):
        return self.VGGLossFactor * mse(self.vgg(fakeImages), self.vgg(high_res))
    
    def adv_loss(self, disc_fake):
        return self.advLossFactor * bce(disc_fake, torch.ones_like(disc_fake))

    def forward(self, fakeImages, high_res, disc_fake):
        return self.content_loss(fakeImages, high_res) + self.adv_loss(disc_fake)

class DiscriminatorLoss(Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, disc_real, disc_fake):
        loss_real = bce(disc_real, torch.ones_like(disc_real))
        loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        return loss_real + loss_fake
