# Example of loading data, normalizing, transforming, and overall preprocessing



import pandas as pd
import torch
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

# drop any rows with na values
df = df.dropna()
df = df.reset_index(drop=True)

# splitting test and training sets
import sklearn
import sklearn.model_selection
df_train, df_test = sklearn.model_selection.train_test_split(
    df, train_size=0.8, random_state=1
)
train_stats = df_train.describe().transpose() # gets descriptive stats (mean, median, etc.) for data, then transpose flips columns and rows with each other


# normalizes the data for these columns
numeric_cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_cols:
    mean = train_stats.loc[col_name, 'mean']
    std  = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = \
        (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = \
        (df_test_norm.loc[:, col_name] - mean)/std
df_train_norm.tail()


# adding the car model years as well, except grouping into sections of years (EX: 0 = before 1973, 1 = 1973 to 1976, etc.)
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)       # for training df
df_train_norm['Model Year Bucketed'] = torch.bucketize(
    v, boundaries, right=True
)

v = torch.tensor(df_test_norm['Model Year'].values)       # for testing df
df_test_norm['Model Year Bucketed'] = torch.bucketize(
    v, boundaries, right=True
)

numeric_cols.append('Model Year Bucketed')


# adding the model year as one-hot encoded, so it'll add more cols to the df
from torch.nn.functional import one_hot
total_origin = len(set(df_train_norm['Origin']))

origin_encoded = one_hot(torch.from_numpy(
    df_train_norm['Origin'].values) % total_origin)   # the % total_origin tells one_hot how many cols it should have, one for each label
x_train_numeric = torch.tensor(
    df_train_norm[numeric_cols].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()  # combines input tensors along the given axis

origin_encoded = one_hot(torch.from_numpy(
    df_test_norm['Origin'].values) % total_origin)
x_test_numeric = torch.tensor(
    df_test_norm[numeric_cols].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()


# label tensors
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()



############################
# Example of creating a NN regression model

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

train_ds = TensorDataset(x_train, y_train)   # premade class that combines labels and data together as long as they are ordered together
batch_size = 8
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# making a model with 2 hidden layers, one with 8 hidden units and the other with 4
hidden_units = [8, 4]
input_size = x_train.shape[1]
all_layers = []
for hidden_unit in hidden_units:                 # creates the model with a for loop, just to have clean code
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 1))  # adds a final Linear layer, which connects the last hidden unit amount to 1 final MPG answer
model = nn.Sequential(*all_layers)  # no clue what the * means


############################
# Example of the training loop to actually train the model



loss_fn = nn.MSELoss()  # good for analyzing continuous data and numbers
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

torch.manual_seed(1)  # don't really need this, just optional
num_epochs = 400
log_epochs = 20
for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
        
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}  Loss '
              f'{loss_hist_train/len(train_dl):.4f}')
        
        
############################
# Example of validating the data with the test datasets

with torch.no_grad():
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
