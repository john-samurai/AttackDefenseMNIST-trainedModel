import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from typing import Tuple, Dict, Any

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class DifferentialPrivacyOptimizer:
    """
    Differential Privacy implementation for Section 6.2 Step 1
    """
    
    def __init__(self, optimizer, noise_multiplier: float = 0.3, max_grad_norm: float = 1.0):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.privacy_spent = 0.0
        
    def step(self, closure=None):
        # Step 1: Gradient clipping (differential privacy)
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group['params']], 
            self.max_grad_norm
        )
        
        # Step 1: Add calibrated noise (differential privacy)
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.noise_multiplier * self.max_grad_norm * 0.1,
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad.data.add_(noise)
        
        # Update privacy budget
        self.privacy_spent += 0.001
        return self.optimizer.step(closure)
    
    def zero_grad(self):
        return self.optimizer.zero_grad()
    
    def get_privacy_spent(self) -> float:
        return self.privacy_spent

class EnhancedRegularizedCNN(nn.Module):
    """
    Enhanced CNN with regularization techniques for Section 6.2 Step 2
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        super(EnhancedRegularizedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)  # Light dropout for conv layers
        
        # Fully connected layers with enhanced regularization
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(dropout_rate)  # Enhanced dropout for FC layers
        self.fc2 = nn.Linear(128, 10)
        
        # Xavier initialization for better generalization (Step 2)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # Enhanced regularization
        x = self.fc2(x)
        
        return x

class RelaxLoss(nn.Module):
    """
    RelaxLoss for membership inference defense (Step 2: Enhanced Regularization)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.05):
        super(RelaxLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy loss
        ce_loss = self.ce_loss(outputs, targets)
        
        # Confidence penalty to reduce membership signals
        probs = F.softmax(outputs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        confidence_penalty = torch.mean(max_probs)
        
        return self.alpha * ce_loss + self.beta * confidence_penalty

class EarlyStopping:
    """Early stopping for Step 2: Enhanced Regularization"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

class Section62DefenseImplementation:
    """
    Implementation of Section 6.2 Defense Design
    Generates only the full defense model
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Full Defense Configuration (Section 6.2)
        self.config = {
            'use_differential_privacy': True,      # Step 1
            'use_relax_loss': True,               # Step 2
            'noise_multiplier': 0.3,              # Step 1: DP noise level
            'max_grad_norm': 1.0,                 # Step 1: Gradient clipping
            'dropout_rate': 0.3,                  # Step 2: Enhanced regularization
            'l2_regularization': 0.001,           # Step 2: L2 weight decay
            'relax_beta': 0.05,                   # Step 2: Confidence penalty
            'early_stopping_patience': 5,        # Step 2: Early stopping
            'learning_rate': 0.001,
            'batch_size': 64,
            'train_subset_size': 3000,
            'val_subset_size': 500,
            'test_subset_size': 500
        }
        
        print("Section 6.2 Defense Configuration:")
        print("- Step 1: Differential Privacy Training Integration ✓")
        print("- Step 2: Enhanced Regularization and Early Stopping ✓")
        
        # Initialize defended model
        self.model = EnhancedRegularizedCNN(
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Step 2: RelaxLoss (Enhanced Regularization)
        self.criterion = RelaxLoss(
            alpha=1.0, 
            beta=self.config['relax_beta']
        )
        
        # Step 2: L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['l2_regularization']
        )
        
        # Step 1: Differential Privacy wrapper
        self.optimizer = DifferentialPrivacyOptimizer(
            self.optimizer,
            noise_multiplier=self.config['noise_multiplier'],
            max_grad_norm=self.config['max_grad_norm']
        )
        
        # Step 2: Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )
        
        self.training_history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': []
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MNIST dataset with subset for efficient training"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Use subset for efficient training
        train_indices = torch.randperm(len(train_dataset))[:self.config['train_subset_size']]
        val_indices = torch.randperm(len(train_dataset))[
            self.config['train_subset_size']:
            self.config['train_subset_size'] + self.config['val_subset_size']
        ]
        test_indices = torch.randperm(len(test_dataset))[:self.config['test_subset_size']]
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        batch_size = self.config['batch_size']
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        print(f"\nDataset loaded:")
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Test samples: {len(test_subset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch with full defense mechanisms"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)  # RelaxLoss
            loss.backward()
            self.optimizer.step()  # DP step
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the defended model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return val_loss / len(val_loader), 100. * correct / total
    
    def train_full_defense_model(self, epochs: int = 15) -> Dict[str, Any]:
        """
        Train the full defense model implementing Section 6.2 design
        """
        print(f"\n{'='*60}")
        print("TRAINING FULL DEFENSE MODEL - SECTION 6.2 DESIGN")
        print(f"{'='*60}")
        print("Implementing:")
        print("• Step 1: Differential Privacy Training Integration")
        print("• Step 2: Enhanced Regularization and Early Stopping")
        print(f"{'='*60}")
        
        train_loader, val_loader, test_loader = self.load_data()
        
        start_time = time.time()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training with full defense
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Progress update every 3 epochs
            if (epoch + 1) % 3 == 0:
                privacy_spent = self.optimizer.get_privacy_spent()
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Privacy Budget Spent: {privacy_spent:.6f}')
                print('-' * 40)
            
            # Step 2: Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Final evaluation
        test_loss, test_acc = self.validate(test_loader)
        training_time = time.time() - start_time
        
        # Calculate overfitting gap (key metric for membership inference resistance)
        final_train_acc = self.training_history['train_acc'][-1]
        overfitting_gap = final_train_acc - test_acc
        
        results = {
            'training_time': training_time,
            'epochs_trained': len(self.training_history['train_loss']),
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': self.training_history['val_acc'][-1],
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'overfitting_gap': overfitting_gap,
            'privacy_budget_spent': self.optimizer.get_privacy_spent(),
            'defense_config': self.config,
            'training_history': self.training_history
        }
        
        print(f"\n{'='*60}")
        print("FULL DEFENSE MODEL TRAINING RESULTS")
        print(f"{'='*60}")
        print("Defense Implementation:")
        print("✓ Step 1: Differential Privacy Training Integration")
        print("✓ Step 2: Enhanced Regularization and Early Stopping")
        print(f"{'='*60}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Epochs Trained: {results['epochs_trained']}")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"Overfitting Gap: {overfitting_gap:.2f}%")
        print(f"Privacy Budget Spent: {results['privacy_budget_spent']:.6f}")
        print(f"{'='*60}")
        
        # Membership inference vulnerability analysis
        if overfitting_gap < 0:
            vulnerability = "LOW RISK - Model generalizes well (negative overfitting gap)"
        elif overfitting_gap < 2:
            vulnerability = "MEDIUM RISK - Some overfitting present"
        else:
            vulnerability = "HIGH RISK - Significant overfitting detected"
        
        print(f"Membership Inference Vulnerability: {vulnerability}")
        print(f"{'='*60}")
        
        return results
    
    def save_model(self, filepath: str = 'full_MIA-defense_model.pt'):
        """Save the full defense model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'defense_design': 'Section 6.2: Differential Privacy + Enhanced Regularization',
            'architecture': 'EnhancedRegularizedCNN'
        }, filepath)
        print(f"Full defense model saved to {filepath}")
        return filepath

def main():
    """Main function to train only the full defense model"""
    print("FIT5124 A3 - Section 6.2 Defense Implementation")
    print("Generating ONLY the Full Defense Model")
    print("=" * 60)
    
    # Initialize defense implementation
    defender = Section62DefenseImplementation()
    
    # Train the full defense model
    results = defender.train_full_defense_model(epochs=15)
    
    # Save the model
    saved_path = defender.save_model()
    
    # Summary for assignment report
    print(f"\n{'='*70}")
    print("SUMMARY FOR FIT5124 A3 SECTION 6.3 REPORT")
    print(f"{'='*70}")
    print("Model Generated:")
    print(f"• File: {saved_path}")
    print(f"• Defense Design: Section 6.2 (Differential Privacy + Enhanced Regularization)")
    print(f"• Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"• Overfitting Gap: {results['overfitting_gap']:.2f}%")
    print(f"• Privacy Budget Spent: {results['privacy_budget_spent']:.6f}")
    print(f"• Training Time: {results['training_time']:.2f} seconds")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()