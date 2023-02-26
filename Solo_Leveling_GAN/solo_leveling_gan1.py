from PIL import Image
import requests
from io import BytesIO
from urllib.request import urlopen, Request

import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#!pip install webdataset
import webdataset as wds
from itertools import islice
SIZE = 256
LENGTH = 20000

preproc = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize(SIZE),
  transforms.Normalize(
      mean=0.5,std=0.5),
])


dataset = (
    wds.WebDataset("./Solo_Leveling_DS.tgz")
    .shuffle(1000)
    .decode("pil", handler=wds.ignore_and_continue)
    .rename(image="jpg;png")
    .map_dict(image=preproc)
    
)
batch_size = 64
# data loader contains values ("key" = link, "image" = actual image tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)


import random

# printing a sample image
for image in islice(dataset, 0, 3):
    plt.imshow((image["image"].permute(1, 2, 0) + 1) / 2)
    
    
# custom weight initialization (for hopefully faster convergence)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
# creating the models
num_filters_d = 64
num_filters_g = 64
input_size = 256
num_end_channels = 3

# the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_size, num_filters_g * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters_g * 32),
            nn.ReLU(True),
            
            # input is 2048 x 8 x 8
            nn.ConvTranspose2d( num_filters_g * 32, num_filters_g * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 16),
            nn.ReLU(True),
            
            # input is 1024 x 16 x 16
            nn.ConvTranspose2d( num_filters_g * 16, num_filters_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 8),
            nn.ReLU(True),

            # input is 512 x 32 x 32
            nn.ConvTranspose2d(num_filters_g * 8, num_filters_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 4),
            nn.ReLU(True),

            # input is 256 x 64 x 64
            nn.ConvTranspose2d( num_filters_g * 4, num_filters_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 2),
            nn.ReLU(True),

            # input is 128 x 128 x 128
            nn.ConvTranspose2d( num_filters_g * 2, num_filters_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g),
            nn.ReLU(True),

            # input is 64 x 256 x 256
            nn.ConvTranspose2d( num_filters_g, num_end_channels, 4, 2, 1, bias=False),
            nn.Tanh()

            # ends as 3 x 512 x 512
        )

    def forward(self, input):
        return self.main(input)

gen_model = Generator().to(device)

from pathlib import Path
path_g = Path('./gen_model_weights.pt')

if (path_g.is_file()) :
    # we can load the model from file to continue training
    gen_model.load_state_dict(torch.load('./gen_model_weights.pt'))
else :
    gen_model.apply(weights_init)
    
    
# the discriminator model

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 512 x 512
            nn.Conv2d(num_end_channels, num_filters_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # input is 64 x 256 x 256
            nn.Conv2d(num_filters_d, num_filters_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # input is 128 x 128 x 128
            nn.Conv2d(num_filters_d * 2, num_filters_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # input is 256 x 64 x 64
            nn.Conv2d(num_filters_d * 4, num_filters_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # input is 512 x 32 x 32
            nn.Conv2d(num_filters_d * 8, num_filters_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # input is 1024 x 16 x 16
            nn.Conv2d(num_filters_d * 16, num_filters_d * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 32),
            nn.LeakyReLU(0.2, inplace=True),

            # input is 2048 x 8 x 8
            nn.Conv2d(num_filters_d * 32, num_filters_d * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            
            
            # input is 4096 x 4 x 4
            nn.Flatten(),
            nn.Linear(num_filters_d * 32 * 8, num_filters_d * 32),
            nn.ReLU(),
            nn.Linear(num_filters_d * 32, num_filters_d * 4),
            nn.ReLU(),
            nn.Linear(num_filters_d * 4, 1),

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

disc_model = Discriminator().to(device)
path_d = Path('./gen_model_weights.pt')

if (path_d.is_file()) :
    # we can load the model from file to continue training
    disc_model.load_state_dict(torch.load('./disc_model_weights.pt'))
else :
    disc_model.apply(weights_init)
 


# loss and optimizers
loss_func = nn.BCELoss()
fixed_z = torch.randn(batch_size, input_size, 1, 1, device=device) # a fixed input vector for the model

from torch import optim
lr = 0.0002
beta1 = 0.5  # don't really know what this does
optim_d = optim.Adam(disc_model.parameters(), lr = lr, betas=(beta1, 0.999))
optim_g = optim.Adam(gen_model.parameters(), lr = lr, betas=(beta1, 0.999))



from torch.cuda import device_of
# Training Loop


# Lists to keep track of progress
sample_list = []
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 10

print("Starting the loop")
for epoch in range(num_epochs):
    # For each batch in the dataloader
    i = 0
    for data in dataloader:

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        disc_model.zero_grad()

        x = data["image"].to(device)
        batch_size = x.size(0)
        label = torch.ones(batch_size, dtype=torch.float, device=device)
        output = disc_model(x).view(-1)
        errD_real = loss_func(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        noise = torch.randn(batch_size, input_size, 1, 1, device=device)
        fake = gen_model(noise)
        
        label.fill_(0)  # makes label a tensor of 0s instead of 1s
        output = disc_model(fake.detach()).view(-1)
        errD_fake = loss_func(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # total loss
        errD = errD_real + errD_fake
        optim_d.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen_model.zero_grad()
        label.fill_(1)  # label now has all 1s again
        output = disc_model(fake).view(-1)
        errG = loss_func(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optim_g.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, int(LENGTH / batch_size),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == int(LENGTH / batch_size)-1)):
            with torch.no_grad():
                fake = gen_model(fixed_z).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            sample_list.append(gen_model(fixed_z).detach().cpu())

        iters += 1
        i += 1
        
        
# displaying images
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())



# showing a sample image from the latest epoch
plt.imshow((sample_list[len(sample_list) - 1][31].permute(1, 2, 0) + 1) / 2)
