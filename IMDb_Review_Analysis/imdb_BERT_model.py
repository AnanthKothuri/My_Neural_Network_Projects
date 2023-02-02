%pip install transformers==4.9.1
import gzip
import shutil
import time
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 3

# importing the imdb movies reviews
url = ("https://github.com/rasbt/"
       "machine-learning-book/raw/"
       "main/ch08/movie_data.csv.gz")
filename = url.split("/")[-1]
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)
with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

df = pd.read_csv('movie_data.csv')

# splitting into test, val, and training sets
train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values
valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values
test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values



# tokenizing the data
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# making a custom Dataset

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        # encodings contain a lot of info, we're taking info of the input_id (number 
        # representation of tokens) and the attention mask (1 or 0). We also add the
        # labels
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()} 
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_ds = IMDbDataset(train_encodings, train_labels)
valid_ds = IMDbDataset(valid_encodings, valid_labels)
test_ds = IMDbDataset(test_encodings, test_labels)

# dataloaders
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=16, shuffle=True
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds, batch_size=16, shuffle=False
)
test_dl = torch.utils.data.DataLoader(
    test_ds, batch_size=16, shuffle=False
)


# Loading BERT model

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()
optim = torch.optim.Adam(model.parameters(), lr = 5e-5)


# calculating accuracy - we have to do it manually to work 
# around RAM and CPU limitations for large models

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            # preparing data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred = (predicted_labels == labels).sum()
            return correct_pred.float()/num_examples * 100
          
  
  
  
# fine-tuning the BERT model; very similar to training a normal Pytorch model

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        # preparing data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        # backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # logging
        if not batch_idx % 250:
            print(f'Epoch: {epoch+1:-4d}/{num_epochs:04d}'
                    f' | Batch'
                    f'{batch_idx:04d}/'
                    f'{len(train_loader):04d} | '
                    f'Loss: {loss:.4f}')      

    model.eval()  
    with torch.set_grad_enabled(False):
        print(f'Training accuracy: '
        f'{compute_accuracy(model, train_loader, device):.2f}%'
        f'\nValid accuracy: '
        f'{compute_accuracy(model, valid_loader, device):.2f}%')

    print(f'Time Elapsed: {(time.time() - start_time)/60:.2f} min')

print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')
