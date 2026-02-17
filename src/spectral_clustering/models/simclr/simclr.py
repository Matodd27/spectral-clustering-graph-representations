import torch
import torch.nn as nn
from torchvision.models import resnet18
from collections import OrderedDict

def get_model(input_channels=1):
    resnet = resnet18(weights=None)
    
    # Modify first layer for 1-channel images (MNIST)
    if input_channels == 1:
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Projection Head
    head = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(resnet.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100,50)),
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
        
    def train(self, batch_size=128, epochs=100, learning_rate=1e-3, log_interval=64):
        
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        
        self.model.train()
        train_loss = 0
        for epoch in range(epochs):
            for batch_idx, (imgs, _) in enumerate(self.dataloaders['train']):
                
                self.optimiser.zero_grad()
                
                x1 = imgs[0].to(device)
                x2 = imgs[1].to(device)
                
                y1 = self.model(x1)
                y2 = self.model(x2)
                
                loss = self.loss_fn(y1, y2)
                train_loss += loss.cpu().data.item()
                
                loss.backward()
                
                self.optimiser.step()
                    
                if batch_idx%log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                        epoch, batch_idx*len(imgs[0]), len(self.dataloaders['train'].dataset),
                        100.*batch_idx / len(self.dataloaders['train']),
                        loss.item() / len(imgs[0])
                    ))   
            print('Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.dataloaders['train'].dataset)
            ))
            
            
        