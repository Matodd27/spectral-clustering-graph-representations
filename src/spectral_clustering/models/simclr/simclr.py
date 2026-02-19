import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_model(input_channels=1, latent_dim=100):
    resnet = resnet18(weights=None)
    
    # Modify first layer for 1-channel images (MNIST)
    if input_channels == 1:
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Projection Head
    head = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, latent_dim)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(latent_dim, 50)),
        ('added_reul2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(50,25))
    ]))

    resnet.fc = head
    return resnet
    
class SimCLR:
    def __init__(self, model, optimiser, dataloaders, loss_fn):
        self.model = model
        self.optimiser = optimiser
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        
    def load_model(self, model_path, remove_top_layers=2):
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if remove_top_layers > 0:
            temp = list(self.model.fc.children())
            if remove_top_layers <= len(temp):
                self.model.fc = torch.nn.Sequential(*temp[:remove_top_layers])
                
    def get_representations(self, mode, device='cpu'):

        self.model.eval()

        res = {
        'X':torch.FloatTensor(),
        'Y':torch.LongTensor()
        }

        with torch.no_grad():
            for batch in self.dataloaders[mode]:

                x = batch[0].to(device)
                label = batch[1]

                pred = self.model(x)

                res['X'] = torch.cat((res['X'], pred.cpu()))
                res['Y'] = torch.cat((res['Y'], label.cpu()))


        res['X'] = np.array(res['X'])
        res['Y'] = np.array(res['Y'])

        return res
        
    def train(self, epochs=100):
        
        losses_train = []
        
        def get_mean_of_list(L):
            return sum(L) / len(L)
        
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        
        self.model.train()
        train_loss = 0
        for epoch in range(epochs):
            epoch_losses_train = []
            for batch_idx, (imgs, _) in enumerate(self.dataloaders['train']):
                
                self.optimiser.zero_grad()
                
                x1 = imgs[0].to(device)
                x2 = imgs[1].to(device)
                
                y1 = self.model(x1)
                y2 = self.model(x2)
                
                loss = self.loss_fn(y1, y2)
                epoch_losses_train.append(loss.cpu().data.item())
                
                loss.backward()
                
                self.optimiser.step()

            losses_train.append(get_mean_of_list(epoch_losses_train))

            # Plot the training losses Graph and save it
            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(losses_train)
            plt.legend(['Training Losses'])
            plt.savefig('losses.png')
            plt.close()

            # Store model and optimizer files
            torch.save(self.model.state_dict(), 'results/model.pth')
            torch.save(self.optimiser.state_dict(), 'results/optimizer.pth')
            np.savez("results/lossesfile", np.array(losses_train))
