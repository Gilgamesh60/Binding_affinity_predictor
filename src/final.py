from torch.utils.data.dataloader import RandomSampler
from data import *
from configure import *
config = Config()
device= config.device
patience = config.patience
dataset = protein_ligand_dataset(config.data_dir, config.affinity_file)
val_size = int(config.val_split * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_size = int(config.test_split * len(val_dataset))
val_size = len(val_dataset) - test_size
val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])
criterion = torch.nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size = config.train_batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = config.val_batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = config.test_batch_size, shuffle = True)
model = GAT(config.in_channels, config.num_gnn_layers, config.num_linear_layers, config.linear_out_channels)
optimizer = Adam(model.parameters(), lr = config.learning_rate)
def train_step(dataloader):
        model.train()
        total_loss = 0.0

        for index, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
            batch = batch.to(device)
            optimizer.zero_grad()

            y_true = batch.y
            y_pred = model(batch)
            y_pred = y_pred.view(y_true.size())

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        return total_loss


def eval_step(dataloader):
        model.eval()
        total_loss = 0.0

        y_trues, y_preds = [], []

        with torch.inference_mode():
            for index, batch in enumerate(tqdm(dataloader, desc = "Iteration")):
                batch = batch.to(device)
                y_true = batch.y
                y_pred = model(batch)
                y_pred = y_pred.view(y_true.size())

                loss = criterion(y_pred, y_true).item()
                total_loss += loss

                y_trues.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())

        y_trues = np.concatenate(y_trues, axis = 0)
        y_preds = np.concatenate(y_preds, axis = 0)

        return total_loss, y_trues, y_preds


best_val_loss = np.inf
best_model = None
train_losses = val_losses = []
for epoch in range(1,Config.num_epochs+1):
  print("---- Epoch {} ----".format(epoch))

  print("Training...")
  train_loss = train_step(train_loader)
  train_losses.append(train_loss)
  print("Evaluating...")
  val_loss, _, _ = eval_step(val_loader)
  val_losses.append(val_loss)

  if scheduler is not None:
    scheduler.step()

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model = model

    current_patience = patience

  else:
    current_patience -= 1

  print("Epoch {} || ".format(epoch),
        "Train Loss: {} || ".format(train_loss),
        "Val Loss: {} || ".format(val_loss))
  if epoch%5==0:
    torch.save(best_model.state_dict(), "/content/drive/MyDrive/exp1/best_model1.pth")
    torch.save(model.state_dict(), "/content/drive/MyDrive/exp1/model1.pth")
    torch.save(optimizer.state_dict(), "/content/drive/MyDrive/exp1/optimizer1.pth")
