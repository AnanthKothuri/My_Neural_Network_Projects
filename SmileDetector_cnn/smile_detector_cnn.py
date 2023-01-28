import torch.nn as nn
import torchvision
from torchvision import transforms
import torch


# doing data augmentation: altering data (flipping, changing saturation, etc.)
# to create more artificial images for training
transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

# used for testing and validation, only resizes image to desired size
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

# this is a lambda function, or a shortened notation for a simple function
get_smile = lambda attr: attr[31]




image_path = './'

celeba_train = torchvision.datasets.CelebA(
    image_path, split='train',
    target_type='attr', download=False,
    transform=transform_train, target_transform=get_smile
)

celeba_valid = torchvision.datasets.CelebA(
    image_path, split='valid',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)

celeba_test = torchvision.datasets.CelebA(
    image_path, split='test',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)



# taking only a small portion of the total dataset
from torch.utils.data import Subset
celeba_train = Subset(celeba_train, torch.arange(16000))
celeba_valid = Subset(celeba_valid, torch.arange(1000))

# data loaders
from torch.utils.data import DataLoader
batch_size = 32
train_dl = DataLoader(celeba_train, 
                      batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid, 
                      batch_size, shuffle=False)
test_dl = DataLoader(celeba_test, 
                      batch_size, shuffle=False)




# making the model

model = nn.Sequential()
model.add_module(
    'conv1', nn.Conv2d(
        in_channels=3, out_channels=32, kernel_size=3, padding=1
    )
)
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.5))


model.add_module(
    'conv2', nn.Conv2d(
        in_channels=32, out_channels=64, kernel_size=3, padding=1
    )
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.5))


model.add_module(
    'conv3', nn.Conv2d(
        in_channels=64, out_channels=128, kernel_size=3, padding=1
    )
)
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))


model.add_module(
    'conv4', nn.Conv2d(
        in_channels=128, out_channels=256, kernel_size=3, padding=1
    )
)
model.add_module('relu4', nn.ReLU())

# global average pool layer: basically average pooling with kernel_size as the
# image size. This averages the value at each channel, reducing the total number 
# of hidden units to flatten for later
model.add_module('pool4', nn.AvgPool2d(kernel_size=8)) # the current image size is 8x8

model.add_module('flatten', nn.Flatten())
model.add_module('l1', nn.Linear(256, 1)) # we have 256 channels after all the convolutions, want to get one output
model.add_module('sigmoid', nn.Sigmoid())






# training the model
loss_fn = nn.BCELoss() # good for binary classification with a single probabilistic output
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
  loss_hist_train = [0] * num_epochs
  loss_hist_valid = [0] * num_epochs
  acc_hist_train = [0] * num_epochs
  acc_hist_valid = [0] * num_epochs

  for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_dl:
      pred = model(x_batch)[:, 0]
      loss = loss_fn(pred, y_batch.float())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      loss_hist_train += loss.item() * y_batch.size(0)
      is_correct = ((pred>0.5).float() == y_batch).float()
      acc_hist_train += is_correct.sum()
    
    loss_hist_train[epoch] /= len(train_dl.dataset)
    acc_hist_train[epoch] /= len(train_dl.dataset)


    model.eval()
    with torch.no_grad():
      for x_batch, y_batch in valid_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch.float())

        loss_hist_train += loss.item() * y_batch.size(0)
        is_correct = ((pred>0.5).float() == y_batch).float()
        acc_hist_train += is_correct.sum()

    loss_hist_valid[epoch] /= len(valid_dl.dataset)
    acc_hist_valid[epoch] /= len(valid_dl.dataset)

    # printing output
    print(f'Epoch {epoch + 1} accuracy: '
          f'{acc_hist_train[epoch]:.4f} val_accuracy: '
          f'{acc_hist_valid[epoch]:.4f}'
    )

  return loss_hist_train, loss_hist_valid, acc_hist_valid, acc_hist_train



# running the training
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)
