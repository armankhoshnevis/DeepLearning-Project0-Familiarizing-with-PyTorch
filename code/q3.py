import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import itertools

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Define model parameters initialization
def init_model_params(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)
        nn.init.zeros_(m.bias)

# ********************************** q3_1 ********************************** #
# Load the data dictionary
data_dict = torch.load('../../HW0_data.pt', map_location="cpu")
x_train = data_dict['x_train']
y_train = data_dict['y_train']
x_val = data_dict['x_val']
y_val = data_dict['y_val']
x_test = data_dict['x_test']

seed_list = [1, 2, 3, 4, 5]
colors = ['red', 'blue', 'purple', 'orange', 'yellow']

plt.figure(figsize=(8, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), color='green',
            marker='o', label='Training Data')

for seed, color in zip(seed_list, colors):
    torch.manual_seed(seed)
    
    # Create the custom dataset
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    
    # Create the dataloaders
    batch_size = 2000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                            shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                            shuffle=False, drop_last=False, pin_memory=True)
    
    # Instantiate the model
    model = MLP()
    
    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the model
    rng = torch.get_rng_state()
    torch.manual_seed(seed)

    model.apply(init_model_params)

    torch.set_rng_state(rng)

    # Define the optimizer
    lr = 1e-2
    weight_decay = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define the loss
    loss_fn = nn.MSELoss()
    
    # Initialize lists
    train_loss_list = []
    val_step_list = []
    val_loss_list = []
    step = 0
    
    # Training loop
    num_epochs = 2000
    for e in trange(num_epochs, desc=f"Training Seed {seed}"):
        model.train()
        for batch in tqdm(train_loader, leave=False, desc="Training"):
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch, y_batch = x_batch.view(-1, 1), y_batch.view(-1, 1)
            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                x_batch, y_batch = x_batch.view(-1, 1), y_batch.view(-1, 1)
                y_hat = model(x_batch)
                loss = loss_fn(y_hat, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_step_list.append(step)
        val_loss_list.append(avg_val_loss)
    
    # Test
    model.eval()
    with torch.no_grad():
        x_test_gpu = x_test.to(device).view(-1, 1)
        y_hat_test = model(x_test_gpu).cpu().numpy()
    
    sorted_indices = torch.argsort(x_test, dim=0).squeeze()
    x_sorted = x_test[sorted_indices].cpu().numpy()
    y_sorted = y_hat_test[sorted_indices]
    plt.plot(x_sorted, y_sorted, linestyle="dotted", color=color, label=f"Seed {seed}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.title("Training Data & Model Predictions for Different Seeds")

plt.savefig("q3_plot.png")

# ********************************** q3_2 ********************************** #
# Define hyperparameter ranges
batch_sizes = [128, 256, 512]
learning_rates = [1e-5, 1e-4, 1e-3]
weight_decays = [1e-4, 1e-3, 1e-2]
seed_list = [1, 2, 3, 4, 5]

# All hyperparameter combinations
hyperparam_grid = list(itertools.product(batch_sizes, learning_rates, weight_decays, seed_list))

# Logging results
results = []

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Iterate over all hyperparameter combinations
best_model = None
best_hyperparams = None
best_val_loss = float("inf")

for batch_size, lr, weight_decay, seed in hyperparam_grid:
    torch.manual_seed(seed)

    # Create datasets & dataloaders
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                            shuffle=False, drop_last=False, pin_memory=True)

    # Instantiate and initialize model
    model = MLP().to(device)
    model.apply(init_model_params)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Training loop
    num_epochs = 200  # Reduce for faster tuning
    for e in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch, y_batch = x_batch.view(-1, 1), y_batch.view(-1, 1)
            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch, y_batch = x_batch.view(-1, 1), y_batch.view(-1, 1)
            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    # Log results
    results.append({"Batch Size": batch_size, "Learning Rate": lr, "Weight Decay": weight_decay, 
                    "Seed": seed, "Validation MSE": avg_val_loss})
    print(f"Seed {seed} | Batch {batch_size}, LR {lr}, WD {weight_decay} → Val MSE: {avg_val_loss:.6f}")
    
    # Track best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_hyperparams = (batch_size, lr, weight_decay)
        best_seed = seed
        best_model = model

# Best hyperparameters
print(f"Best Hyperparameters: Batch {best_hyperparams[0]}, LR {best_hyperparams[1]}, WD {best_hyperparams[2]}, Seed {best_seed}")

# Test dataset
test_dataset = CustomDataset(x_test)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0,
                         shuffle=False, drop_last=False, pin_memory=True)

# Best model and test dataset
best_model.eval()
with torch.no_grad():
    for batch in test_loader:
        x_test_gpu = batch.to(device).view(-1, 1)
        y_hat_test = best_model(x_test_gpu).cpu().numpy()

# Save predictions
with open("q3_test_output.txt", "w") as f:
    for yhat in y_hat_test:
        f.write(f"{yhat.item()}\n")

# Save the best model
torch.save(best_model.state_dict(), "q3_model.pt")

print("\n Predictions saved in `q3_test_output.txt`!")
