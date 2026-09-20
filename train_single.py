import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def parse_args():
    p = argparse.ArgumentParser(description='MNIST model training')
    p.add_argument('--datadir', type=str, default='../data/', )
    p.add_argument('--epochs', type=int, default=20,)
    p.add_argument('--batch_size', type=int, default=64, )
    p.add_argument('--compile', action="store_true")
    p.add_argument('--lr', type=float, default=3e-4, )
    p.add_argument('--num_workers', type=int, default=8, )
    p.add_argument('--eval_freq', type=int, default=4,)
    

    return p.parse_args()

def getloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.datadir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.datadir, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    return train_loader, test_loader

def train(model, train_loader, optimizer, epoch, device, scheduler, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}]'
                  f'loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}')
    
def test(model, test_loader, epoch, device):
    model.eval()
    accuracy = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            predictions = torch.argmax(torch.nn.Softmax(dim=1)(output), dim=1)
            predictions == target
            running_accuracy = torch.sum(predictions == target) / len(predictions)
            running_accuracy = running_accuracy.item()
            accuracy += running_accuracy
        
    print(f'epoch: {epoch} | running accuracy: {accuracy/(batch_idx+1):.2f}')
         
def main(args):
    train_loader, test_loader = getloader(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    warmup_steps = int(0.05 * args.epochs * len(train_loader))
    warmup = LinearLR(
        optimizer,
        start_factor = 0.01,
        end_factor = 1,
        total_iters = warmup_steps
        )
    
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs*len(train_loader) - warmup_steps,
        eta_min = 3e-6,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )
    print('Training starts here.......')
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, epoch, device, scheduler)
        if epoch % args.eval_freq == 0:
            test(model, test_loader, epoch, device)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)