import torchvision
import torch
from torchvision import transforms
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=True
)

from torch.utils.data import Subset
mnist_valid = Subset(mnist_dataset,
                     torch.arange(10000))
mnist_train = Subset(mnist_dataset,
                    torch.arange(10000, len(mnist_dataset)
                    ))
mnist_test = torchvision.datasets.MNIST(
    root=image_path, train=False,
    transform=transform, download=True
)

from torch.utils.data import DataLoader
batch_size = 64
train_dl = DataLoader(mnist_train, 
                     batch_size, 
                     shuffle=True)
valid_dl = DataLoader(mnist_valid,
                      batch_size,
                      shuffle=False)



#################################
# making the model

import torch.nn as nn
model = nn.Sequential()

model.add_module('conv1', 
                 nn.Conv2d(
                     in_channels=1, out_channels=32,
                     kernel_size=5, padding=2))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))  # stride for maxpool is set to default as kernel_size as well, unless otherwise specified



model.add_module('conv2', 
                 nn.Conv2d(
                     in_channels=32, out_channels=64,
                     kernel_size=5, padding=2))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('flatten', nn.Flatten())

# fully connected layers
model.add_module('l1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))  # dropout layer in between, 0.5 chance of dropping each weight, makes model more robust
model.add_module('l2', nn.Linear(1024, 10))  # bring it down to final 10 options


# would normally add a SoftMax layer as well to get probabilities of last 10 options, but
# the CrossEntropyLoss automatically does this for us


#############################

# training the model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    acc_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    acc_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        model.train()   # sets model to training mode
        
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (
                torch.argmax(pred, dim=1) == y_batch
            ).float()
            acc_hist_train[epoch] += is_correct.sum()
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        acc_hist_train[epoch] /= len(train_dl.dataset)
        
        # evaluating accuracy of model
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (
                    torch.argmax(pred, dim=1) == y_batch
                ).float()
                acc_hist_valid[epoch] += is_correct.sum()
        
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        acc_hist_valid[epoch] /= len(valid_dl.dataset)
        
        
        # printing info
        print(f'Epoch {epoch + 1} accuracy: '
             f'{acc_hist_train[epoch]:.4f} val_accuracy: '
             f'{acc_hist_valid[epoch]:.4f}')
        
    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid
  
  
##########################
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)
