# Creating a GAN model for the MNIST hand-written numbers dataset

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# generator model
def make_generator_network(input_size, n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4,
                           1, 0, bias=False),
        nn.BatchNorm2d(n_filters*4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*4, n_filters*2,
                           3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters*2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*2, n_filters,
                           4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1,
                           bias=False),
        nn.Tanh()
    )
    return model

# discriminator model
class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters*2,
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4,
                      3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(0)

# initializing models
z_size = 100
image_size = (28, 28)
n_filters = 32
gen_model = make_generator_network(z_size, n_filters).to(device)

disc_model = Discriminator(n_filters).to(device)

# preparing MNIST data
# since generator network will produce pixel values of (-1, 1) because of Tanh function,
# we have to adjust out input data to match that

import torchvision
from torchvision import transforms
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5)),
])
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=True
)


from torch.utils.data import DataLoader
batch_size = 64
mnist_dl = DataLoader(mnist_dataset, batch_size=batch_size,
                      shuffle=True, drop_last=True)

# getting a random vector z based on the desired destribution
def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size, 1, 1)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size, 1, 1)

    return input_z
  

loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen_model.parameters())
d_optimizer = torch.optim.Adam(disc_model.parameters())
mode_z = 'uniform'


# training the discriminator
def d_train(x):
    disc_model.zero_grad()

    # train the discriminator with a real batch
    batch_size = x.size(0)
    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = disc_model(x) # predicted output
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    # train the discriminator with a fake batch
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z) # fake image created by generator model
    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)

    # optimize loss and gradient for discriminator only
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()


# training the generator
def g_train(x):
    gen_model.zero_grad()
    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones(batch_size, 1, device=device)

    g_output = gen_model(input_z) # the generated image
    d_proba_fake = disc_model(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    # optimize loss and gradient for generator only
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.data.item()
# the training loop
fixed_z = create_noise(batch_size, z_size, mode_z).to(device)

# returns a generated image
def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0 # transforms pixels to the range (0, 1) instead of (-1, 1)

epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100

for epoch in range(1, num_epochs+1):
    gen_model.train()
    d_losses, g_losses = [], []
    d_vals_real, d_vals_fake = [], []
    for i, (x, _) in enumerate(mnist_dl):
        d_loss, d_proba_real, d_proba_fake = d_train(x)
        d_losses.append(d_loss)
        g_losses.append(g_train(x)) # train the generator model
        # d_vals_real.append(d_proba_real.mean().cpu())
        # d_vals_fake.append(d_proba_fake.mean().cpu())

    # all_d_losses.append(torch.tensor(d_losses).mean())
    # all_g_losses.append(torch.tensor(g_losses).mean())
    # all_d_real.append(torch.tensor(d_vals_real).mean())
    # all_d_fake.append(torch.tensor(d_vals_fake).mean())

    print(f'Epoch {epoch:03d} | Avg Losses >>'
            f' G/D {torch.FloatTensor(g_losses).mean():.4f}'
            f'/{torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples.append(
        create_samples(
            gen_model, fixed_z
        ).detach().cpu().numpy()
    )
 
# saving both model parameters
torch.save(gen_model.state_dict(), "gen_model_weights.pt")
torch.save(gen_model, "gen_model.pt")
torch.save(disc_model.state_dict(), "disc_model_weights.pt")
torch.save(disc_model, "disc_model.pt")

# graphing the loss for both models over epochs

import itertools
fig = plt.figure(figsize=(16, 6))
# plotting losses
ax = fig.add_subplot(1, 2, 1)
plt.plot(all_g_losses, label='Generator Loss')
half_d_losses = [all_d_loss/2 for all_d_loss in all_d_losses]
plt.plot(half_d_losses, label='Discriminator Loss')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)

# plotting outputs of the discriminator stats
ax = fig.add_subplot(1, 2, 2)
plt.plot(all_d_real, label=r'Real: $D(mathbf{x})$')
plt.plot(all_d_fake, label=r'Fake: $D(G(mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator Output', size=15)
plt.show()

selected_epochs = [1, 2, 4, 10, 25, 50, 100]
fig = plt.figure(figsize=(10, 14))
for i, e in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(7, 5, i*5+j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j==0:
            ax.text(
                -0.06, 0.5, f'Epoch {e}',
                rotation=90, size=18, color='red', 
                horizontalalignment='right', 
                verticalalignment='center',
                transform=ax.transAxes
            )

        image = epoch_samples[e-1][j]
        ax.imshow(image, cmap='gray_r')

plt.show()
