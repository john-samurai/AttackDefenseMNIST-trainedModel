# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


class MinimalOverfitModel(nn.Module):
    def __init__(self):
        super(MinimalOverfitModel, self).__init__()
        # Extremely simple model - just one conv layer and FC layers
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2, padding=2)  # 28->14
        # After conv1: 28x28 -> 14x14, after maxpool: 7x7
        # So: 8 * 7 * 7 = 392
        self.fc1 = nn.Linear(8 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 14->7
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Use simpler transform - NO normalization to make learning easier
transform = transforms.Compose([
    transforms.ToTensor()
])
    
# Load dataset
dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)

# Create VERY small training set - even smaller!
train_size = 50  # Even smaller!
unused_size = len(dataset1) - train_size

# Set seed for reproducibility
torch.manual_seed(42)
train_subset, _ = torch.utils.data.random_split(dataset1, [train_size, unused_size])

# Use batch size = 1 for maximum overfitting
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000, shuffle=False)

# Build the minimal model
model = MinimalOverfitModel()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(train_subset)}")
print(f"Parameters per sample ratio: {sum(p.numel() for p in model.parameters()) / len(train_subset):.1f}")

# Use Adam with moderate learning rate first
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

print("Starting aggressive overfitting training...")
print("Goal: Force the model to memorize the small training set")

best_train_acc = 0
patience = 50
no_improvement = 0

for epoch in range(1, 1001):  # More epochs
    epoch_loss = 0
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Check progress every 10 epochs at first, then less frequently
    if epoch <= 100 and epoch % 10 == 0:
        check_progress = True
    elif epoch <= 200 and epoch % 20 == 0:
        check_progress = True
    elif epoch % 50 == 0:
        check_progress = True
    else:
        check_progress = False
        
    if check_progress:
        model.eval()
        
        # Training accuracy - check all training samples
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
        
        # Test accuracy - just first 1000 samples
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
                break  # Only first batch
        
        test_accuracy = 100. * test_correct / test_total
        gap = train_accuracy - test_accuracy
        
        print(f'Epoch {epoch}: Loss: {avg_epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Gap: {gap:.2f}%')
        
        # Track improvement
        if train_accuracy > best_train_acc + 1:  # Significant improvement
            best_train_acc = train_accuracy
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Dynamic adjustments based on progress
        if epoch == 100 and train_accuracy < 40:
            print("Learning too slow after 100 epochs, increasing learning rate")
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            
        elif epoch == 200 and train_accuracy < 60:
            print("Still too slow, trying SGD with high learning rate")
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            
        elif epoch == 300 and train_accuracy < 70:
            print("Trying even higher learning rate")
            optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
            
        elif epoch == 400 and train_accuracy < 80:
            print("Final attempt with very high learning rate")
            optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
        
        # Success conditions
        if train_accuracy >= 98:
            print(f"Near perfect memorization achieved at epoch {epoch}!")
            break
        elif train_accuracy >= 90 and gap > 10:
            print(f"Good overfitting achieved at epoch {epoch}!")
            break

print(f"\nTraining completed. Best training accuracy: {best_train_acc:.2f}%")

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
train_total = 0
model.eval()
with torch.no_grad():
    for data, target in train_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += target.size(0)

train_accuracy = 100. * train_correct / train_total
test_accuracy = 100. * correct / len(test_loader.dataset)
overfitting_gap = train_accuracy - test_accuracy

print('\n' + '='*60)
print('AGGRESSIVE OVERFITTING MODEL TRAINING COMPLETE')
print('='*60)
print(f'Training Set Size: {len(train_subset)} samples')
print(f'Model Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Parameters per Sample: {sum(p.numel() for p in model.parameters()) / len(train_subset):.1f}')
print(f'Training Accuracy: {train_accuracy:.2f}%')
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Overfitting Gap: {overfitting_gap:.2f}%')
print(f'Final Test Loss: {test_loss:.4f}')

if overfitting_gap > 10 and train_accuracy > 90:
    print('SUCCESS: Model shows STRONG overfitting!')
elif overfitting_gap > 5 and train_accuracy > 80:
    print('GOOD: Model shows moderate overfitting')
elif train_accuracy > 70:
    print('MODERATE: Model learned training data reasonably well')
else:
    print('POOR: Model failed to memorize training data')
    print('The dataset might be too difficult or model too simple')

# Save the overfitted model
torch.save(model.state_dict(), "mnist_cnn_overfitted.pt")
print(f'\nModel saved as: mnist_cnn_overfitted.pt')

# Enhanced confidence score analysis
print('\n' + '='*60)
print('CONFIDENCE SCORE ANALYSIS')
print('='*60)

model.eval()
with torch.no_grad():
    # Training samples confidence analysis
    train_confidences = []
    train_entropies = []
    train_max_probs = []
    
    for data, target in train_loader:
        output = model(data)
        probs = torch.exp(output)
        
        # Max probability
        max_probs = torch.max(probs, dim=1)[0]
        train_max_probs.extend(max_probs.tolist())
        
        # Entropy
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        train_entropies.extend(entropies.tolist())
        
        # Average confidence
        avg_confidences = torch.mean(probs, dim=1)
        train_confidences.extend(avg_confidences.tolist())
    
    # Test samples confidence analysis  
    test_confidences = []
    test_entropies = []
    test_max_probs = []
    
    for data, target in test_loader:
        output = model(data)
        probs = torch.exp(output)
        
        # Max probability
        max_probs = torch.max(probs, dim=1)[0]
        test_max_probs.extend(max_probs.tolist())
        
        # Entropy
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        test_entropies.extend(entropies.tolist())
        
        # Average confidence
        avg_confidences = torch.mean(probs, dim=1)
        test_confidences.extend(avg_confidences.tolist())
        break  # Just first batch for efficiency

# Calculate statistics
train_conf_mean = np.mean(train_confidences)
test_conf_mean = np.mean(test_confidences)
conf_diff = train_conf_mean - test_conf_mean

train_maxprob_mean = np.mean(train_max_probs)
test_maxprob_mean = np.mean(test_max_probs)
maxprob_diff = train_maxprob_mean - test_maxprob_mean

train_entropy_mean = np.mean(train_entropies)
test_entropy_mean = np.mean(test_entropies)
entropy_diff = train_entropy_mean - test_entropy_mean

print(f'Training samples ({len(train_confidences)} samples):')
print(f'  Avg confidence: {train_conf_mean:.4f}')
print(f'  Avg max prob: {train_maxprob_mean:.4f}')
print(f'  Avg entropy: {train_entropy_mean:.4f}')

print(f'\nTest samples ({len(test_confidences)} samples):')
print(f'  Avg confidence: {test_conf_mean:.4f}')
print(f'  Avg max prob: {test_maxprob_mean:.4f}')
print(f'  Avg entropy: {test_entropy_mean:.4f}')

print(f'\nDifferences (Train - Test):')
print(f'  Confidence difference: {conf_diff:.4f}')
print(f'  Max prob difference: {maxprob_diff:.4f}')
print(f'  Entropy difference: {entropy_diff:.4f}')

# Overall assessment
signals = 0
if abs(conf_diff) > 0.01:
    signals += 1
    print('✓ Confidence difference detected')
if abs(maxprob_diff) > 0.05:
    signals += 1
    print('✓ Max probability difference detected')
if abs(entropy_diff) > 0.1:
    signals += 1
    print('✓ Entropy difference detected')

if signals >= 2:
    print('\nSUCCESS: Multiple signals indicate good potential for MIA!')
elif signals == 1:
    print('\nMODERATE: Some signals detected for MIA')
else:
    print('\nWARNING: Few distinguishing signals - MIA may be difficult')

print(f'\nModel ready for membership inference attack!')

# Debug: Show some sample predictions
print('\n' + '='*40)
print('SAMPLE TRAINING PREDICTIONS:')
print('='*40)
model.eval()
sample_count = 0
with torch.no_grad():
    for data, target in train_loader:
        if sample_count >= 5:
            break
        output = model(data)
        probs = torch.exp(output)
        pred = output.argmax(dim=1)
        max_prob = torch.max(probs, dim=1)[0]
        
        print(f'True: {target.item()}, Pred: {pred.item()}, Confidence: {max_prob.item():.3f}')
        sample_count += 1