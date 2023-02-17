import torchvision.datasets as datasets
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from pathlib import Path

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([64, 64]),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])

# importing plant datasets
car_ds = datasets.StanfordCars(root='./',
                                split='train',
                                transform=transform,
                                download=True)
batch_size = 64
# dataloader
from torch.utils.data import DataLoader
car_dl = DataLoader(car_ds, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom weight initialization (for hopefully faster convergence)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
num_filters_d = 64
num_filters_g = 64
input_size = 100
num_end_channels = 3

# the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_size, num_filters_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(num_filters_g * 8, num_filters_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d( num_filters_g * 4, num_filters_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( num_filters_g * 2, num_filters_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_g),
            nn.ReLU(True),

            nn.ConvTranspose2d( num_filters_g, num_end_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)

path_g = Path('./gen_model_weights.pt')
gen_model = Generator().to(device)

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
            # input is 3 x 64 x 64
            nn.Conv2d(num_end_channels, num_filters_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters_d, num_filters_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters_d * 2, num_filters_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters_d * 4, num_filters_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

disc_model = Discriminator().to(device)
path_g = Path('./disc_model_weights.pt')

if (path_g.is_file()) :
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
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 15

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(car_dl, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        disc_model.zero_grad()

        x = data[0].to(device)
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
                  % (epoch, num_epochs, i, len(car_dl),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(car_dl)-1)):
            with torch.no_grad():
                fake = gen_model(fixed_z).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        
     
torch.save(gen_model.state_dict(), "gen_model_weights.pt")
torch.save(gen_model, "gen_model.pt")
torch.save(disc_model.state_dict(), "disc_model_weights.pt")
torch.save(disc_model, "disc_model.pt")


# graphing loss vs training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
