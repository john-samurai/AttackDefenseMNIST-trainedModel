# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        # Remove dropout layers to encourage overfitting
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # Remove dropout during forward pass
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

#Define normalization 
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
#Load dataset
dataset1 = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('./data', train=False,
                   transform=transform)

# Create smaller training set to encourage overfitting (use only 10% of training data)
train_size = int(0.1 * len(dataset1))
unused_size = len(dataset1) - train_size
train_subset, _ = torch.utils.data.random_split(dataset1, [train_size, unused_size])

# Use smaller batch size for more frequent updates
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)

#Build the model we defined above
model = Lenet()

# Use higher learning rate to encourage overfitting
optimizer = optim.Adadelta(model.parameters(), lr=2.0)
# Remove scheduler to maintain high learning rate
# scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

model.train()
# Train for more epochs to encourage overfitting
for epoch in range(1, 31):  # Increased from 5 to 30 epochs
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Calculate and print epoch-level training accuracy
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for data, target in train_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
    
    train_accuracy = 100. * train_correct / train_total
    avg_epoch_loss = epoch_loss / len(train_loader)
    
    # Calculate test accuracy
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
    
    test_accuracy = 100. * test_correct / test_total
    gap = train_accuracy - test_accuracy
    
    print(f'Epoch {epoch}: Train Loss: {avg_epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Gap: {gap:.2f}%')
    
    model.train()
    # Don't use scheduler to maintain high learning rate
    # scheduler.step()

# Final evaluation
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

# Calculate final training accuracy
train_correct = 0
model.eval()
with torch.no_grad():
    for data, target in train_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()

train_accuracy = 100. * train_correct / len(train_loader.dataset)
test_accuracy = 100. * correct / len(test_loader.dataset)
overfitting_gap = train_accuracy - test_accuracy

print('\n' + '='*60)
print('OVERFITTING MODEL TRAINING COMPLETE')
print('='*60)
print(f'Training Set Size: {len(train_loader.dataset)} samples ({train_size/len(dataset1)*100:.1f}% of original)')
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Overfitting Gap: {overfitting_gap:.2f}%')
print(f'Final Test Loss: {test_loss:.4f}')

if overfitting_gap > 10:
    print('SUCCESS: Model shows significant overfitting (gap > 10%)')
else:
    print('WARNING: Model may not be sufficiently overfitted')

# Save the overfitted model
torch.save(model.state_dict(), "mnist_cnn_overfitted.pt")
print(f'\nOverfitted model saved as: overfitting_mnist_cnn.pt')
print('This model can be used for membership inference attack experiments.')