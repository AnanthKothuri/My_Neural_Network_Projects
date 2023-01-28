%pip install torchtext
%pip install torchdata
import torch
from torchtext.datasets import IMDB

train_ds = IMDB(split='train')
test_ds = IMDB(split='test')

from torch.utils.data.dataset import random_split
train_ds, valid_ds = random_split(list(train_ds), [20000, 5000])

from torch.utils.data import Subset
train_ds = Subset(train_ds, torch.arange(2000))
valid_ds = Subset(valid_ds, torch.arange(500))
test_ds = Subset(test_ds, torch.arange(1000))


from torch.utils.data import DataLoader
# finding unique tokes (words)
import re
import torch.nn as nn
import torch
from collections import Counter, OrderedDict

# method to filter movie reviews to only get words, remove emojis, etc.
def tokenizer(text):
  text = re.sub('<[^>]*>', '', text)
  emoticons = re.findall(
      '(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower()
  )
  text = re.sub('[\W]+', ' ', text.lower()) +\
      ' '.join(emoticons).replace('-', '')
  tokenized = text.split()
  return tokenized

# stores the unique tokens and their frequencies (which we don't really need for this project)
token_counts = Counter()
for label, line in train_ds:
    tokens = tokenizer(line)
    token_counts.update(tokens)


# encoding each token into unique integers
from torchtext.vocab import vocab
sorted_by_freq = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)   # sorts the tokens based on frequency
ordered_dict = OrderedDict(sorted_by_freq)
vocab = vocab(ordered_dict)
vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1) # for tokens not in training ds, but are found in testing or validation
vocab.set_default_index(1)



# converting labels to 1 (positive review) or 0 (negative review)
label_pipeline = lambda x: 1 if x == 2 else 0
# converting text review to number representations of words
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]


# performs the preprocessing on the data for the batch
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)  # pads the text so that each list has the same length

    return padded_text_list, label_list, lengths


# dataloaders
batch_size = 32
train_dl = DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_ds, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_batch)


# building the model

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, l_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim,
                                      padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.l1 = nn.Linear(rnn_hidden_size, l_hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(l_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)  # rnn gives the output AND the hidden values for the next hidden layers
        out = hidden[-1, :, :]
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return out

      
 # training the model

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
l_hidden_size = 64

model = RNN(vocab_size, embed_dim, rnn_hidden_size, l_hidden_size)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

model


# method that trains for one epoch
def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch.float())
        loss.backward()
        optimizer.step()

        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)
    
    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)

# method that evaluates for an epoch
def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch.float())

            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)
  
# running the model

num_epochs = 10
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f}'
            f' val_accuracy: {acc_valid:.4f}')
