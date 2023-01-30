import numpy as np
import torch
import torch.nn as nn


# using Jules Verne's Mysterious Island to train this model
with open('macbeth.txt', 'r', encoding="utf8") as f:
    text = f.read()
start_ndx = text.find('Actus Primus')
end_ndx = text.find('FINIS')
text = text[start_ndx:end_ndx]
char_set = set(text)


# mapping characters to ints, to be used in training
char_sorted = sorted(char_set)
char2int = {ch:i for i, ch in enumerate(char_sorted)}
char_array = np.array(char_sorted)
text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32
) # we've now encoded the entire book into numbers


# building the datasets for the model

from torch.utils.data import Dataset, DataLoader
seq_length = 80
chunk_size = seq_length + 1
text_chunks = [text_encoded[i: i+chunk_size] 
               for i in range(len(text_encoded) - chunk_size+1)]

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()  # the input (x) is the first 40 characters
                                                              # the target (y) is the last 40 characters
        

batch_size = 64
seq_ds = TextDataset(torch.tensor(text_chunks))
seq_dl = DataLoader(seq_ds, batch_size=batch_size,
                    shuffle=True, drop_last=True)

# creating the model

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.l1 = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))  # ensuring that the previous hidden layers + the new input 
                                                             # contribute to the new output 
        out = self.l1(out).reshape(out.size(0), -1)
        return out, hidden, cell   # doesn't change 'out' to probability, leaves it as logits, this will transform back into a letter

    def init_hidden(self, batch_size):  # gets the hidden, cell for the very first element, which doesn't have any preceding hidden, cell values
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
      
# training

vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512

model = RNN(vocab_size, embed_dim, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 3000
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))   # only using one batch from the dataloader, but have a lot of epochs
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model (seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 100 == 0:
        print(f'Epoch {epoch} loss: {loss:.4f}')

# evaluating, generating new text

from torch.distributions.categorical import Categorical

# takes a starting text, generates new characters and appends
def sample(model, starting_str, len_generated_text=500, scale_factor=1):
    encoded_input = torch.tensor(
        [char2int[s] for s in starting_str]
    )
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

    return generated_str
  
# running the model
print(sample(model, starting_str="Death", len_generated_text=1000, scale_factor=3))
