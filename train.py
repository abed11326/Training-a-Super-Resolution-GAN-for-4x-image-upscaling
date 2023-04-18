from models import Generator, Discriminator
from objectives import GeneratorLoss, DiscriminatorLoss, mse
from imageData import ImageData
from torch.utils.data import DataLoader
from hypParam import *
from torch import sigmoid
from torch.optim import Adam
from evaluation import evaluate
from statistics import mean

# initializations
gen = Generator(no_resBlocks).to(device)
disc = Discriminator().to(device)
mse_opt = Adam(gen.parameters(), mse_lr)
gen_opt = Adam(gen.parameters(), initial_lr)
disc_opt = Adam(disc.parameters(), initial_lr)
gen_loss_fn = GeneratorLoss(advLossFactor, VGGLossFactor)
disc_loss_fn = DiscriminatorLoss()

dataset = ImageData(train_data_path, HR_shape)
loader = DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=no_workers)

# pretrain with SRResnet
for epoch in range(1, mse_epochs+1):
    for low_res, high_res in loader:
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        resnet_out = gen(low_res)
        loss = mse(resnet_out, high_res)
        mse_opt.zero_grad()
        loss.backward()
        mse_opt.step()

    if epoch % 70 == 0:
        evaluate(gen, 10, f'./results/mse/epoch_{epoch}', 'MSE-SRResNet')
        print(f"Finished: {epoch}")

#save SRResNet's parameters
torch.save(gen.state_dict(), './parameters/SRResNet_param.pt')
print("Saved MSE-SRResNet Parameters")
print("Finished: MSE-SRResNet")

# train SRGAN 
gen.load_state_dict(torch.load('./parameters/SRResNet_param.pt'))
print("Loaded the pretrained SRResNet weights")

for epoch in range(1, gan_epochs+1):
    gen_loss_hist = []
    disc_loss_hist = []
    disc_real_hist = []
    disc_fake_hist = []

    for low_res, high_res in loader:
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        # train discriminator
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss = disc_loss_fn(disc_real, disc_fake)
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
        # train generator
        disc_fake = disc(fake)
        gen_loss = gen_loss_fn(fake, high_res, disc_fake)
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        gen_loss_hist.append(gen_loss.item())
        disc_loss_hist.append(disc_loss.item())
        disc_real_hist.append(sigmoid(disc_real).mean().item())
        disc_fake_hist.append(sigmoid(disc_fake).mean().item())

    print(f"Epoch: {epoch}, Gen Loss: {round(mean(gen_loss_hist), 4)}, Disc Loss: {round(mean(disc_loss_hist), 4)}, disc_real: {round(mean(disc_real_hist), 4)}, disc_fake: {round(mean(disc_fake_hist), 4)}")

    if epoch % 20 == 0:
        evaluate(gen, 10, f'./results/srgan/epoch_{epoch}', 'SRGAN')
        print(f"Finished: {epoch}")

    if epoch == gan_epochs / 2:
        #decrease lr
        for group in gen_opt.param_groups:
            group['lr'] = second_lr
        for group in disc_opt.param_groups:
            group['lr'] = second_lr
        print("Decreased LR")

torch.save(gen.state_dict(), './parameters/SRGAN_gen_param.pt')
print("Saved SRGAN Gen Parameters")
torch.save(disc.state_dict(), './parameters/SRGAN_disc_param.pt')
print("Saved SRGAN Disc Parameters")
print("Finished: SRGAN")