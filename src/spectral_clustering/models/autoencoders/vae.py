from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn
from torch.nn import functional as f
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def vae(data, layer_widths=[400,20], batch_size=128, epochs=100, learning_rate=1e-3, save_interval=None, save_path=None, epoch_start=0):
    class InputDataset(Dataset):
        def __init__(self, data, targets, transform=None):
            self.data = data
            self.targets = targets
            self.transform = transform
            
        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            
            if self.transform:
                x = self.transform(x)
                
            return x,y
        
        def __len__(self):
            return len(self.data)
    
    # Structure as defined in Calder (2020) [https://github.com/GraphLearning/]
    class VAE(nn.Module):
        def __init__(self, layer_widths):
            super(VAE, self).__init__()
            
            self.lw = layer_widths
            self.fc1 = nn.Linear(self.lw[0], self.lw[1])
            self.fc21 = nn.Linear(self.lw[1], self.lw[2])
            self.fc22 = nn.Linear(self.lw[1], self.lw[2])
            self.fc3 = nn.Linear(self.lw[2], self.lw[1])
            self.fc4 = nn.Linear(self.lw[1], self.lw[0])
            
        def encode(self, x):
            h1 = f.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)
        
        def reparameterise(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
            
        def decode(self, z):
            h3 = f.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))
        
        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.lw[0]))
            z = self.reparameterise(mu, logvar)
            return self.decode(z), mu, logvar
        
    def beta_vae_loss(x_hat, x, mu, logvar, beta=1):
        
        BCE = f.binary_cross_entropy(x_hat, x.view(-1, data.shape[1]), reduction='sum')
        
        # Kingma and Welling (2014)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta*KLD
    
        
    def train(epoch, losses_train=None):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            loss = beta_vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(data_loader.dataset),
                    100.*batch_idx / len(data_loader),
                    loss.item() / len(data)
                ))         
            
        print('Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_loader.dataset)
        ))
        if losses_train is not None:
            losses_train.append(train_loss / len(data_loader.dataset))
            
    layer_widths = [data.shape[1]] + layer_widths
    log_interval = 64
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    data = data - data.min()
    data = data/data.max()
    data = torch.from_numpy(data).float()
    target = np.zeros((data.shape[0],)).astype(int)
    target = torch.from_numpy(target).long()
    dataset = InputDataset(data, target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    model = VAE(layer_widths).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if (save_interval is not None) and (save_path is not None):
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if os.path.isfile(f"{save_path}/epoch_{epoch_start}.pt"):
            model.load_state_dict(torch.load(f"{save_path}/epoch_{epoch_start}.pt"))
            optimizer.load_state_dict(torch.load(f"{save_path}/optimizer_epoch_{epoch_start}.pt"))
            
            temp = np.load(f"{save_path}/lossesfile.npz")
            losses_train = list(temp['arr_0'])[:epoch_start+1]
        else:
            losses_train = []
            
        for epoch in range(1, epochs+1):
            train(epoch+epoch_start, losses_train=losses_train)
            if epoch % save_interval == 0:
                torch.save(model.state_dict(), f"{save_path}/epoch_{epoch+epoch_start}.pt")
                torch.save(optimizer.state_dict(), f"{save_path}/optimizer_epoch_{epoch+epoch_start}.pt")
                np.savez(f"{save_path}/lossesfile.npz", losses_train)
                
                fig = plt.figure()
                sns.set_style('darkgrid')
                plt.plot(losses_train)
                plt.legend(['Training Losses'])
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('VAE Training Losses')
                plt.savefig(f"{save_path}/losses_plot.pdf", bbox_inches="tight")
                plt.savefig(f"{save_path}/losses_plot.png", bbox_inches="tight")
                plt.close()
    else:
        for epoch in range(1, epochs+1):
            train(epoch)
        
    with torch.no_grad():
        mu, logvar = model.encode(data.to(device).view(-1, layer_widths[0]))
        data_vae = mu.cpu().numpy()
        
    return data_vae