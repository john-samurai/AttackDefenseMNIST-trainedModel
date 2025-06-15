import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class OriginalTargetModel(nn.Module):
    """
    Original architecture (matches actual mnist_cnn.pt)
    Based on the error messages, the actual architecture is:
    - conv1: (1, 6, 3, 3) 
    - conv2: (16, 6, 5, 5)
    - fc1: (120, 400)
    - fc2: (84, 120) 
    - fc3: (10, 84)
    """
    def __init__(self):
        super(OriginalTargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(400, 120)  # 16 * 5 * 5 = 400 after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DefendedTargetModel(nn.Module):
    """
    Defended model architecture (matches full_MIA-defense_model.pt)
    """
    def __init__(self, dropout_rate=0.3):
        super(DefendedTargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class ShadowModel(nn.Module):
    """
    Shadow model for generating attack training data
    """
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class AttackModel(nn.Module):
    """
    Binary classifier for membership inference
    """
    def __init__(self, input_size=10):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class ComparativeMembershipInferenceAttack:
    """
    Membership inference attack comparing original vs defended models
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.original_model = None
        self.defended_model = None
        self.shadow_models = []
        self.attack_model = AttackModel()
        
        # Load MNIST dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Results storage
        self.results = {
            'original_model': {},
            'defended_model': {},
            'comparison': {}
        }
        
    def load_models(self):
        """Load both original and defended models"""
        print("="*60)
        print("LOADING TARGET MODELS FOR COMPARISON")
        print("="*60)
        
        # Load original model
        print("1. Loading Original Model (mnist_cnn.pt)...")
        self.original_model = OriginalTargetModel().to(self.device)
        try:
            checkpoint = torch.load('mnist_cnn.pt', map_location=self.device)
            self.original_model.load_state_dict(checkpoint)
            self.original_model.eval()
            print("✅ Original model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading original model: {e}")
            return False
        
        # Load defended model
        print("\n2. Loading Defended Model (full_MIA-defense_model.pt)...")
        self.defended_model = DefendedTargetModel(dropout_rate=0.3).to(self.device)
        try:
            checkpoint = torch.load('full_MIA-defense_model.pt', map_location=self.device)
            # Handle different possible checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.defended_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.defended_model.load_state_dict(checkpoint)
            self.defended_model.eval()
            print("✅ Defended model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading defended model: {e}")
            return False
        
        print("\n✅ Both models loaded successfully!")
        return True
    
    def evaluate_model_performance(self):
        """Evaluate basic performance of both models"""
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*60)
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluate original model
        original_accuracy = self._evaluate_single_model(self.original_model, test_loader, "Original Model")
        
        # Evaluate defended model
        defended_accuracy = self._evaluate_single_model(self.defended_model, test_loader, "Defended Model")
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"Original Model Accuracy: {original_accuracy:.2f}%")
        print(f"Defended Model Accuracy: {defended_accuracy:.2f}%")
        print(f"Accuracy Drop due to Defense: {original_accuracy - defended_accuracy:.2f}%")
        
        self.results['comparison']['original_accuracy'] = original_accuracy
        self.results['comparison']['defended_accuracy'] = defended_accuracy
        self.results['comparison']['accuracy_drop'] = original_accuracy - defended_accuracy
        
        return original_accuracy, defended_accuracy
    
    def _evaluate_single_model(self, model, test_loader, model_name):
        """Evaluate a single model's accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"{model_name} Accuracy: {accuracy:.2f}%")
        return accuracy
    
    def prepare_shadow_data(self, num_models=3, samples_per_model=1000):
        """Prepare shadow datasets for attack training"""
        print(f"\n" + "="*60)
        print("PREPARING SHADOW DATASETS")
        print("="*60)
        print(f"Creating {num_models} shadow models with {samples_per_model} samples each...")
        
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        shadow_datasets = []
        for i in range(num_models):
            indices = torch.randperm(len(full_dataset))[:samples_per_model]
            shadow_data = torch.utils.data.Subset(full_dataset, indices)
            shadow_datasets.append(shadow_data)
            print(f"Shadow dataset {i+1}: {len(shadow_data)} samples")
        
        return shadow_datasets
    
    def train_shadow_models(self, shadow_datasets):
        """Train shadow models for attack data generation"""
        print(f"\n" + "="*60)
        print("TRAINING SHADOW MODELS")
        print("="*60)
        
        self.shadow_models = []
        
        for i, shadow_data in enumerate(shadow_datasets):
            print(f"\nTraining Shadow Model {i+1}...")
            
            # Split shadow data
            train_size = int(0.5 * len(shadow_data))
            test_size = len(shadow_data) - train_size
            train_data, test_data = torch.utils.data.random_split(shadow_data, [train_size, test_size])
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            
            # Train shadow model
            shadow_model = ShadowModel().to(self.device)
            optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
            
            shadow_model.train()
            for epoch in range(10):
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = shadow_model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 3 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"  Epoch {epoch+1}/10, Average Loss: {avg_loss:.4f}")
            
            shadow_model.eval()
            self.shadow_models.append((shadow_model, train_data, test_data))
        
        print("✅ Shadow models training completed!")
    
    def generate_attack_training_data(self):
        """Generate training data for the attack model"""
        print(f"\n" + "="*60)
        print("GENERATING ATTACK TRAINING DATA")
        print("="*60)
        
        attack_features = []
        attack_labels = []
        
        for i, (shadow_model, train_data, test_data) in enumerate(self.shadow_models):
            print(f"Processing Shadow Model {i+1}...")
            
            # Member samples (from training data)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
            member_count = 0
            
            with torch.no_grad():
                for data, _ in train_loader:
                    if member_count >= 200:
                        break
                    
                    data = data.to(self.device)
                    output = shadow_model(data)
                    confidence_scores = torch.exp(output).cpu().numpy().flatten()
                    
                    attack_features.append(confidence_scores)
                    attack_labels.append(1)  # Member
                    member_count += 1
            
            # Non-member samples (from test data)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
            nonmember_count = 0
            
            with torch.no_grad():
                for data, _ in test_loader:
                    if nonmember_count >= 200:
                        break
                    
                    data = data.to(self.device)
                    output = shadow_model(data)
                    confidence_scores = torch.exp(output).cpu().numpy().flatten()
                    
                    attack_features.append(confidence_scores)
                    attack_labels.append(0)  # Non-member
                    nonmember_count += 1
        
        attack_features = np.array(attack_features)
        attack_labels = np.array(attack_labels)
        
        print(f"Generated {len(attack_features)} training samples")
        print(f"Members: {np.sum(attack_labels)}, Non-members: {len(attack_labels) - np.sum(attack_labels)}")
        
        return attack_features, attack_labels
    
    def train_attack_model(self, attack_features, attack_labels):
        """Train the membership inference attack model"""
        print(f"\n" + "="*60)
        print("TRAINING ATTACK MODEL")
        print("="*60)
        
        # Convert to tensors
        X = torch.FloatTensor(attack_features).to(self.device)
        y = torch.FloatTensor(attack_labels).unsqueeze(1).to(self.device)
        
        # Split into train/validation
        train_size = int(0.8 * len(X))
        indices = torch.randperm(len(X))
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize attack model
        self.attack_model = AttackModel().to(self.device)
        optimizer = optim.Adam(self.attack_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Train attack model
        self.attack_model.train()
        for epoch in range(50):
            epoch_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                predictions = self.attack_model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                self.attack_model.eval()
                with torch.no_grad():
                    val_predictions = self.attack_model(X_val)
                    val_loss = criterion(val_predictions, y_val)
                    val_accuracy = ((val_predictions > 0.5).float() == y_val).float().mean()
                    print(f"Epoch {epoch+1}/50 - Train Loss: {epoch_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                self.attack_model.train()
        
        self.attack_model.eval()
        print("✅ Attack model training completed!")
    
    def perform_membership_inference_on_model(self, target_model, model_name):
        """Perform membership inference attack on a specific model"""
        print(f"\n" + "="*50)
        print(f"ATTACKING {model_name.upper()}")
        print("="*50)
        
        # Create test dataset with known membership
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=self.transform)
        
        # Create member and non-member samples
        member_indices = torch.randperm(len(test_dataset))[:500]
        member_samples = torch.utils.data.Subset(test_dataset, member_indices)
        
        all_indices = set(range(len(test_dataset)))
        member_indices_set = set(member_indices.tolist())
        nonmember_indices = list(all_indices - member_indices_set)[:500]
        nonmember_samples = torch.utils.data.Subset(test_dataset, nonmember_indices)
        
        # Combine samples and create labels
        combined_samples = []
        combined_labels = []
        
        for data, label in member_samples:
            combined_samples.append((data, label))
            combined_labels.append(1)  # Member
        
        for data, label in nonmember_samples:
            combined_samples.append((data, label))
            combined_labels.append(0)  # Non-member
        
        # Perform inference
        start_time = time.time()
        predicted_memberships = []
        confidence_patterns = []
        
        test_loader = torch.utils.data.DataLoader(combined_samples, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for (data, _), true_label in zip(test_loader, combined_labels):
                data = data.to(self.device)
                
                # Get model confidence scores
                target_output = target_model(data)
                if model_name == "Original Model":
                    # For original model, apply softmax to get probabilities
                    confidence_scores = F.softmax(target_output, dim=1).cpu().numpy().flatten()
                else:
                    # For defended model, apply softmax to get probabilities
                    confidence_scores = F.softmax(target_output, dim=1).cpu().numpy().flatten()
                
                confidence_patterns.append(confidence_scores)
                
                # Use attack model to predict membership
                attack_input = torch.FloatTensor(confidence_scores).unsqueeze(0).to(self.device)
                membership_prob = self.attack_model(attack_input).item()
                predicted_memberships.append(membership_prob)
        
        inference_time = time.time() - start_time
        
        # Convert predictions to binary
        binary_predictions = [1 if prob > 0.5 else 0 for prob in predicted_memberships]
        
        # Calculate metrics
        accuracy = accuracy_score(combined_labels, binary_predictions)
        
        # Calculate confidence statistics
        member_confidences = [confidence_patterns[i] for i in range(len(confidence_patterns)) if combined_labels[i] == 1]
        nonmember_confidences = [confidence_patterns[i] for i in range(len(confidence_patterns)) if combined_labels[i] == 0]
        
        member_max_conf = np.mean([np.max(conf) for conf in member_confidences])
        nonmember_max_conf = np.mean([np.max(conf) for conf in nonmember_confidences])
        confidence_gap = member_max_conf - nonmember_max_conf
        
        # Print results
        print(f"Attack Results for {model_name}:")
        print(f"├─ Total test samples: {len(combined_samples)}")
        print(f"├─ Attack accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"├─ Inference time: {inference_time:.4f} seconds")
        print(f"├─ Member avg confidence: {member_max_conf:.4f}")
        print(f"├─ Non-member avg confidence: {nonmember_max_conf:.4f}")
        print(f"└─ Confidence gap: {confidence_gap:.4f}")
        
        # Store results
        results = {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'member_avg_confidence': member_max_conf,
            'nonmember_avg_confidence': nonmember_max_conf,
            'confidence_gap': confidence_gap,
            'predictions': binary_predictions,
            'membership_probabilities': predicted_memberships
        }
        
        return results
    
    def comparative_evaluation(self):
        """Perform comparative membership inference evaluation"""
        print("\n" + "="*70)
        print("COMPARATIVE MEMBERSHIP INFERENCE ATTACK EVALUATION")
        print("="*70)
        
        # Setup attack components
        shadow_datasets = self.prepare_shadow_data(num_models=3, samples_per_model=1000)
        self.train_shadow_models(shadow_datasets)
        attack_features, attack_labels = self.generate_attack_training_data()
        self.train_attack_model(attack_features, attack_labels)
        
        # Attack both models
        print("\n" + "="*70)
        print("PERFORMING ATTACKS ON BOTH MODELS")
        print("="*70)
        
        original_results = self.perform_membership_inference_on_model(self.original_model, "Original Model")
        defended_results = self.perform_membership_inference_on_model(self.defended_model, "Defended Model")
        
        # Store results
        self.results['original_model'] = original_results
        self.results['defended_model'] = defended_results
        
        # Calculate defense effectiveness
        attack_reduction = original_results['accuracy'] - defended_results['accuracy']
        defense_effectiveness = (attack_reduction / original_results['accuracy']) * 100 if original_results['accuracy'] > 0 else 0
        
        self.results['comparison']['attack_accuracy_reduction'] = attack_reduction
        self.results['comparison']['defense_effectiveness'] = defense_effectiveness
        
        return self.results
    
    def print_final_comparison(self):
        """Print comprehensive comparison results"""
        print("\n" + "="*80)
        print("FINAL COMPARATIVE ANALYSIS - SECTION 6.3 RESULTS")
        print("="*80)
        
        print("\n📊 MODEL PERFORMANCE COMPARISON:")
        print(f"├─ Original Model Accuracy: {self.results['comparison']['original_accuracy']:.2f}%")
        print(f"├─ Defended Model Accuracy: {self.results['comparison']['defended_accuracy']:.2f}%")
        print(f"└─ Utility Loss: {self.results['comparison']['accuracy_drop']:.2f}%")
        
        print("\n🔒 PRIVACY PROTECTION ANALYSIS:")
        original_attack_acc = self.results['original_model']['accuracy']
        defended_attack_acc = self.results['defended_model']['accuracy']
        
        print(f"├─ Attack Success on Original Model: {original_attack_acc:.4f} ({original_attack_acc*100:.2f}%)")
        print(f"├─ Attack Success on Defended Model: {defended_attack_acc:.4f} ({defended_attack_acc*100:.2f}%)")
        print(f"├─ Attack Success Reduction: {self.results['comparison']['attack_accuracy_reduction']:.4f} ({self.results['comparison']['attack_accuracy_reduction']*100:.2f}%)")
        print(f"└─ Defense Effectiveness: {self.results['comparison']['defense_effectiveness']:.2f}%")
        
        print("\n🎯 CONFIDENCE PATTERN ANALYSIS:")
        print(f"Original Model:")
        print(f"├─ Member confidence: {self.results['original_model']['member_avg_confidence']:.4f}")
        print(f"├─ Non-member confidence: {self.results['original_model']['nonmember_avg_confidence']:.4f}")
        print(f"└─ Confidence gap: {self.results['original_model']['confidence_gap']:.4f}")
        
        print(f"\nDefended Model:")
        print(f"├─ Member confidence: {self.results['defended_model']['member_avg_confidence']:.4f}")
        print(f"├─ Non-member confidence: {self.results['defended_model']['nonmember_avg_confidence']:.4f}")
        print(f"└─ Confidence gap: {self.results['defended_model']['confidence_gap']:.4f}")
        
        print("\n🏆 DEFENSE ASSESSMENT:")
        if defended_attack_acc < 0.55:
            assessment = "EXCELLENT - Attack success near random guessing"
        elif defended_attack_acc < 0.65:
            assessment = "GOOD - Significant attack mitigation"
        elif defended_attack_acc < 0.75:
            assessment = "MODERATE - Some privacy protection"
        else:
            assessment = "WEAK - Limited privacy protection"
        
        print(f"Defense Quality: {assessment}")
        
        print("\n" + "="*80)
        print("SECTION 6.3 IMPLEMENTATION AND EVALUATION COMPLETE")
        print("="*80)

def main():
    """Main function for comparative membership inference evaluation"""
    print("="*80)
    print("COMPARATIVE MEMBERSHIP INFERENCE ATTACK EVALUATION")
    print("FIT5124 A3 - Section 6.3 Implementation and Evaluation")
    print("="*80)
    print("Comparing privacy vulnerability: Original vs Defended models")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize comparative attack
    attack = ComparativeMembershipInferenceAttack(device=device)
    
    # Load both models
    if not attack.load_models():
        print("❌ Failed to load models. Please ensure both models are available.")
        return
    
    # Evaluate basic model performance
    attack.evaluate_model_performance()
    
    # Perform comparative membership inference evaluation
    results = attack.comparative_evaluation()
    
    # Print final comparison
    attack.print_final_comparison()
    
    print("\n✅ Evaluation completed! Use these results for your Section 6.3 report.")
    
    return results

if __name__ == "__main__":
    main()