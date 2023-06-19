from torch_geometric.data import DataLoader
import warnings
from configure import *
from data import *
warnings.filterwarnings("ignore")
# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
dataset = protein_ligand_dataset(Config.root,Config.data_dir,Config.affinity_file)
# Use cuda or CPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GAT()
model = model.to(device)
# Wrap data in a data loader
data_size = len(dataset)
NUM_GRAPHS_PER_BATCH = Config.graph_batch
train_loader = DataLoader(dataset[:int(data_size * 0.8)],
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):],
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(data):
    # Enumerate over the data
    for batch in train_loader:
      # Use GPU
      batch.to(device)
      # Reset gradients
      optimizer.zero_grad()
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
      # Calculating the loss and gradients
      loss = loss_fn(pred, batch.y)
      loss.backward()
      # Update using the gradients
      optimizer.step()
    return loss, embedding

print("Starting training...")
losses = []
for epoch in range(Config.epochs):
    loss, h = train(dataset)
    losses.append(loss)
    if epoch%Config.epoch_step%1=0:
      print(f"Epoch {epoch} | Train Loss {loss}")
print("Traininig complete successfully!")

print("Saving the model...")

torch.save(model,'model.pt')
torch.save(optimizer,'optimizer.pt')

print("Model saved")
