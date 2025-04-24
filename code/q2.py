import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from data import MyDataset

# Set seeds
overall_seed = 34916
model_seed = 916
torch.manual_seed(overall_seed)

# Generate the dataset
alpha = 2
beta = 0.5
num_samples = 2000

train_dataset = MyDataset(alpha, beta, num_samples, 'train', overall_seed)
val_dataset = MyDataset(alpha, beta, num_samples, 'val', overall_seed)

# Create the dataloaders
batch_size = num_samples
num_workers = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers, pin_memory=True)

# Create a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Initialize the model
rng = torch.get_rng_state()  # Store current random state (overall seed)
torch.manual_seed(model_seed)  # Set seed for model initialization (model seed)

nn.init.normal_(model.linear.weight)
nn.init.zeros_(model.linear.bias)

torch.set_rng_state(rng)  # Restore original random state (again, back to overall seed)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create an AdamW optimizer
lr = 1e-2
weight_decay = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Set up the loss function
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 1000

train_step_list = []
train_loss_list = []
val_step_list = []
val_loss_list = []
w_list = []
step = 0

for e in trange(num_epochs, desc='Epochs'):
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_loader, leave=False, desc='Training'):
        # Zero gradients
        optimizer.zero_grad()
        
        # Unpack the batch and move to GPU
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass (model prediction)
        y_hat = model(x_batch)
        
        # Calculate the loss
        loss = loss_fn(y_hat, y_batch)
        total_train_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
        train_step_list.append(step)
        train_loss_list.append(loss.item())
        step += 1
        
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            y_hat = model(x_batch)
            
            loss = loss_fn(y_hat, y_batch)
            total_val_loss += loss.item()
    
    val_step_list.append(step)
    val_loss_list.append(total_val_loss / len(val_loader))
    
    if e % 100 == 0:
        print(f"Epoch {e+1}/{num_epochs}, Train Loss: {total_train_loss / len(train_loader):.6f}, Validation Loss: {total_val_loss / len(val_loader):.6f}")

# Final results
print("True parameters: alpha =", alpha, "beta =", beta)
print("Estimated parameters: alpha =", model.linear.weight.item(), "beta =", model.linear.bias.item())
print("Prediction error on validation set:", total_val_loss / len(val_loader))

# Plot the training and validation loss
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(train_step_list, train_loss_list, label='Train Loss')
axs[0].plot(val_step_list, val_loss_list, label='Validation Loss')
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].set_title('Loss vs Step')
axs[0].set_yscale('log')
axs[0].legend()

# Plot the Data and Fitted Model
val_x = val_dataset.x
val_y = val_dataset.y

x_min = val_x.min() - 0.2 * val_x.min().abs()
x_max = val_x.max() + 0.2 * val_x.max().abs()
x_test = torch.linspace(x_min, x_max, 1000).view(-1, 1).to(device)

model.eval()
with torch.no_grad():
    y_hat = model(x_test).cpu().numpy()

axs[1].scatter(val_x, val_y, label='Data', c="blue", marker=".")
axs[1].plot(x_test.cpu(), y_hat, label='Fitted Model', c="red")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].grid(True)
axs[1].set_title('Data and Fitted Model')
axs[1].legend()

fig.tight_layout()
fig.savefig('../results/q2_plot.png', dpi=300)
plt.clf()
plt.close(fig)

# Save the Model
torch.save(model.state_dict(), '../results/q2_model.pt')