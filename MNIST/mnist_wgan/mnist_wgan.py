# Creating a GAN model for the MNIST hand-written numbers dataset

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# generator model - NOTE: WGAN models work better with InstanceNorm instead of BatchNorm
def make_generator_network(input_size, n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4,
                           1, 0, bias=False),
        nn.InstanceNorm2d(n_filters*4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*4, n_filters*2,
                           3, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters*2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters*2, n_filters,
                           4, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters),
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
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters*2, n_filters*4,
                      3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters*4),
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
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)
mode_z = 'uniform'
lambda_gp = 10.0


# training the discriminator
def d_train_wgan(x):
    disc_model.zero_grad()

    # train the discriminator with a real batch
    batch_size = x.size(0)
    x = x.to(device)
    
    d_real = disc_model(x)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z) # fake image created by generator model
    d_generated = disc_model(g_output)

    # optimize loss and gradient for discriminator only
    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output.data)
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()


# training the generator
def g_train_wgan(x):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z) # the generated image

    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()

    # optimize loss and gradient for generator only
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss.data.item()

# calculates the gradient penalty (like our own custom loss functions for WGANs)
from torch.autograd import grad as torch_grad
def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1,
                       requires_grad=True, device=device)
    interpolated = alpha * real_data + \
                   (1 - alpha) * generated_data

    # Calculate probability of interpolated examples
    proba_interpolated = disc_model(interpolated)

    # Calculate gradients of probabilities
    gradients = torch_grad(
        outputs=proba_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(proba_interpolated.size(),
                                device=device),
        create_graph=True, retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1)**2).mean()
  
# the training loop
fixed_z = create_noise(batch_size, z_size, mode_z).to(device)

# returns a generated image
def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0 # transforms pixels to the range (0, 1) instead of (-1, 1)

epoch_samples_wgan = []
num_epochs = 100
critic_iterations = 5

for epoch in range(1, num_epochs+1):
    gen_model.train()

    d_losses, g_losses = [], []
    d_vals_real, d_vals_fake = [], []
    for i, (x, _) in enumerate(mnist_dl):
        for _ in range(critic_iterations):
            d_loss = d_train_wgan(x)
        d_losses.append(d_loss)
        g_losses.append(g_train_wgan(x)) # train the generator model

    print(f'Epoch {epoch:03d} | D Loss >>'
            f' {torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples_wgan.append(
        create_samples(
            gen_model, fixed_z
        ).detach().cpu().numpy()
    )
    
# saving both model parameters
torch.save(gen_model.state_dict(), "gen_wgan_weights.pt")
torch.save(gen_model, "gen_wgan.pt")
torch.save(disc_model.state_dict(), "disc_wgan_weights.pt")
torch.save(disc_model, "disc_wgan.pt")

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

        image = epoch_samples_wgan[e-1][j]
        ax.imshow(image, cmap='gray_r')

plt.show()
