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

class TargetModel(nn.Module):
    """
    Target model architecture (LeNet-style) that we want to attack
    This simulates the bank's check processing model
    """
    def __init__(self):
        super(TargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
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

class ShadowModel(nn.Module):
    """
    Shadow model with similar architecture to target model
    Used to simulate target model behavior for generating attack training data
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
    Binary classifier to perform membership inference
    Input: confidence scores from target/shadow models (10 features)
    Output: membership probability (member=1, non-member=0)
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

class EnhancedMembershipInferenceAttack:
    """
    Enhanced class implementing the membership inference attack with configurable parameters
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.target_model = None
        self.shadow_models = []
        self.attack_model = AttackModel()
        
        # Load MNIST dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Timing statistics
        self.timing_stats = {
            'shadow_training_time': 0,
            'attack_training_time': 0,
            'inference_time': 0
        }
        
    def load_target_model(self, model_path):
        """Load the pre-trained target model (bank's check processing model)"""
        print("Loading target model (Bank's Check Processing Model)...")
        self.target_model = TargetModel()
        try:
            self.target_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_model.eval()
            print("Target model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading target model: {e}")
            print("Training a new target model for demonstration...")
            return self._train_target_model()
    
    def _train_target_model(self):
        """Train a target model if pre-trained model is not available"""
        print("Training target model...")
        
        # Load training data
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        # Use subset for overfitting (as mentioned in your scenario)
        train_size = int(0.1 * len(train_dataset))  # Use 10% for overfitting
        train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
        
        self.target_model = TargetModel().to(self.device)
        optimizer = optim.Adam(self.target_model.parameters(), lr=0.001)
        
        self.target_model.train()
        for epoch in range(20):  # More epochs for overfitting
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.target_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Target Model - Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        self.target_model.eval()
        
        # Save the trained model
        torch.save(self.target_model.state_dict(), 'target_model_overfitted.pt')
        print("Target model training completed and saved!")
        return True
    
    def prepare_shadow_data(self, num_models=3, samples_per_model=1000):
        """
        Step 1: Shadow Dataset Collection and Preparation
        Create shadow datasets from MNIST for training shadow models
        """
        print(f"\nStep 1: Preparing shadow datasets for {num_models} shadow models...")
        print(f"Each shadow model will use {samples_per_model} samples")
        
        # Load full MNIST dataset
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        shadow_datasets = []
        
        for i in range(num_models):
            # Create random subset for each shadow model
            indices = torch.randperm(len(full_dataset))[:samples_per_model]
            shadow_data = torch.utils.data.Subset(full_dataset, indices)
            shadow_datasets.append(shadow_data)
            print(f"Shadow dataset {i+1}: {len(shadow_data)} samples")
        
        return shadow_datasets
    
    def train_shadow_models(self, shadow_datasets):
        """
        Step 2: Shadow Model Training
        Train multiple shadow models to simulate target model behavior
        """
        print(f"\nStep 2: Training {len(shadow_datasets)} shadow models...")
        start_time = time.time()
        
        self.shadow_models = []
        
        for i, shadow_data in enumerate(shadow_datasets):
            print(f"\nTraining Shadow Model {i+1}...")
            
            # Split shadow data into train/test for member/non-member labels
            train_size = int(0.5 * len(shadow_data))
            test_size = len(shadow_data) - train_size
            train_data, test_data = torch.utils.data.random_split(shadow_data, [train_size, test_size])
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
            
            # Initialize shadow model
            shadow_model = ShadowModel().to(self.device)
            optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
            
            # Train shadow model
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
        
        self.timing_stats['shadow_training_time'] = time.time() - start_time
        print(f"Shadow models training completed in {self.timing_stats['shadow_training_time']:.2f} seconds!")
    
    def generate_attack_training_data(self, samples_per_class=200):
        """
        Step 3: Generate training data for attack model using shadow models
        Creates member/non-member confidence score pairs
        """
        print(f"\nStep 3: Generating attack model training data...")
        print(f"Using {samples_per_class} samples per class (member/non-member) per shadow model")
        
        attack_features = []
        attack_labels = []
        
        for i, (shadow_model, train_data, test_data) in enumerate(self.shadow_models):
            print(f"Processing Shadow Model {i+1}...")
            
            # Generate member samples (from training data)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
            member_count = 0
            
            with torch.no_grad():
                for data, _ in train_loader:
                    if member_count >= samples_per_class:
                        break
                    
                    data = data.to(self.device)
                    output = shadow_model(data)
                    confidence_scores = torch.exp(output).cpu().numpy().flatten()
                    
                    attack_features.append(confidence_scores)
                    attack_labels.append(1)  # Member
                    member_count += 1
            
            # Generate non-member samples (from test data)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
            nonmember_count = 0
            
            with torch.no_grad():
                for data, _ in test_loader:
                    if nonmember_count >= samples_per_class:
                        break
                    
                    data = data.to(self.device)
                    output = shadow_model(data)
                    confidence_scores = torch.exp(output).cpu().numpy().flatten()
                    
                    attack_features.append(confidence_scores)
                    attack_labels.append(0)  # Non-member
                    nonmember_count += 1
        
        attack_features = np.array(attack_features)
        attack_labels = np.array(attack_labels)
        
        print(f"Generated {len(attack_features)} training samples for attack model")
        print(f"Members: {np.sum(attack_labels)}, Non-members: {len(attack_labels) - np.sum(attack_labels)}")
        
        return attack_features, attack_labels
    
    def train_attack_model(self, attack_features, attack_labels):
        """
        Step 3 (continued): Train the attack model for membership inference
        """
        print("\nTraining Attack Model...")
        start_time = time.time()
        
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
            
            # Validation
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
        self.timing_stats['attack_training_time'] = time.time() - start_time
        print(f"Attack model training completed in {self.timing_stats['attack_training_time']:.2f} seconds!")
    
    def perform_membership_inference(self, test_samples, true_membership_labels, target_name="Target Model"):
        """
        Step 4: Perform membership inference attack on target model
        """
        print(f"\nStep 4: Performing membership inference attack on {target_name}...")
        start_time = time.time()
        
        # Get confidence scores from target model
        predicted_memberships = []
        confidence_scores_list = []
        
        test_loader = torch.utils.data.DataLoader(test_samples, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                
                # Get target model confidence scores
                target_output = self.target_model(data)
                confidence_scores = torch.exp(target_output).cpu().numpy().flatten()
                confidence_scores_list.append(confidence_scores)
                
                # Use attack model to predict membership
                attack_input = torch.FloatTensor(confidence_scores).unsqueeze(0).to(self.device)
                membership_prob = self.attack_model(attack_input).item()
                predicted_memberships.append(membership_prob)
        
        # Convert predictions to binary (threshold = 0.5)
        binary_predictions = [1 if prob > 0.5 else 0 for prob in predicted_memberships]
        
        # Calculate metrics
        accuracy = accuracy_score(true_membership_labels, binary_predictions)
        
        self.timing_stats['inference_time'] = time.time() - start_time
        
        # Print results
        print(f"\n=== MEMBERSHIP INFERENCE ATTACK RESULTS ===")
        print(f"Target: {target_name}")
        print(f"Total test samples: {len(test_samples)}")
        print(f"Actual members: {sum(true_membership_labels)}")
        print(f"Actual non-members: {len(true_membership_labels) - sum(true_membership_labels)}")
        print(f"Predicted members: {sum(binary_predictions)}")
        print(f"Attack Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Inference time: {self.timing_stats['inference_time']:.4f} seconds")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(true_membership_labels, binary_predictions, 
                                  target_names=['Non-Member', 'Member']))
        
        return {
            'accuracy': accuracy,
            'predictions': binary_predictions,
            'membership_probabilities': predicted_memberships,
            'confidence_scores': confidence_scores_list
        }
    
    def evaluate_attack_effectiveness(self, num_shadow_models=3, samples_per_shadow_model=1000, 
                                    samples_per_class=200):
        """
        Comprehensive evaluation of the membership inference attack with configurable parameters
        """
        print("\n" + "="*60)
        print("ENHANCED MEMBERSHIP INFERENCE ATTACK EVALUATION")
        print("="*60)
        print(f"Configuration:")
        print(f"- Number of shadow models: {num_shadow_models}")
        print(f"- Samples per shadow model: {samples_per_shadow_model}")
        print(f"- Training samples per class: {samples_per_class}")
        print("="*60)
        
        # Load test dataset
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=self.transform)
        
        # Create member and non-member test sets
        # Members: samples that were potentially in training
        member_indices = torch.randperm(len(test_dataset))[:500]
        member_samples = torch.utils.data.Subset(test_dataset, member_indices)
        
        # Non-members: remaining samples
        all_indices = set(range(len(test_dataset)))
        member_indices_set = set(member_indices.tolist())
        nonmember_indices = list(all_indices - member_indices_set)[:500]
        nonmember_samples = torch.utils.data.Subset(test_dataset, nonmember_indices)
        
        # Combine and create labels
        combined_samples = []
        combined_labels = []
        
        # Add member samples
        for data, label in member_samples:
            combined_samples.append((data, label))
            combined_labels.append(1)  # Member
        
        # Add non-member samples  
        for data, label in nonmember_samples:
            combined_samples.append((data, label))
            combined_labels.append(0)  # Non-member
        
        # Execute attack with custom parameters
        shadow_datasets = self.prepare_shadow_data(num_shadow_models, samples_per_shadow_model)
        self.train_shadow_models(shadow_datasets)
        
        attack_features, attack_labels = self.generate_attack_training_data(samples_per_class)
        self.train_attack_model(attack_features, attack_labels)
        
        # Perform inference attack
        results = self.perform_membership_inference(combined_samples, combined_labels)
        
        # Print timing summary
        print(f"\n=== TIMING ANALYSIS ===")
        print(f"Shadow Models Training: {self.timing_stats['shadow_training_time']:.2f} seconds")
        print(f"Attack Model Training: {self.timing_stats['attack_training_time']:.2f} seconds")
        print(f"Membership Inference: {self.timing_stats['inference_time']:.4f} seconds")
        print(f"Total Attack Time: {sum(self.timing_stats.values()):.2f} seconds")
        
        return results

def run_parameter_experiments():
    """
    Run experiments with different parameter configurations
    """
    print("="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define experiment configurations
    experiments = [
        # (num_shadow_models, samples_per_shadow_model, samples_per_class)
        (3, 1000, 200),   # Original configuration
        (5, 1000, 200),   # More shadow models
        (10, 1000, 200),  # Even more shadow models
        (3, 2000, 200),   # More data per shadow model
        (3, 1000, 500),   # More training samples per class
        (5, 2000, 500),   # Combined improvements
    ]
    
    results_summary = []
    
    for i, (num_shadows, samples_per_shadow, samples_per_class) in enumerate(experiments):
        print(f"\n{'='*50}")
        print(f"EXPERIMENT {i+1}: {num_shadows} shadows, {samples_per_shadow} samples/shadow, {samples_per_class} samples/class")
        print(f"{'='*50}")
        
        # Initialize fresh attack instance for each experiment
        attack = EnhancedMembershipInferenceAttack(device=device)
        
        # Load target model
        target_loaded = attack.load_target_model('mnist_cnn.pt')
        if not target_loaded:
            print("Failed to load target model. Skipping experiment...")
            continue
        
        # Run experiment
        try:
            results = attack.evaluate_attack_effectiveness(
                num_shadow_models=num_shadows,
                samples_per_shadow_model=samples_per_shadow,
                samples_per_class=samples_per_class
            )
            
            results_summary.append({
                'config': f"{num_shadows}s-{samples_per_shadow}d-{samples_per_class}c",
                'accuracy': results['accuracy'],
                'total_time': sum(attack.timing_stats.values())
            })
            
        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Configuration':<15} {'Accuracy':<10} {'Total Time (s)':<15}")
    print("-" * 40)
    
    for result in results_summary:
        print(f"{result['config']:<15} {result['accuracy']:<10.4f} {result['total_time']:<15.2f}")
    
    return results_summary

def main():
    """
    Main function to demonstrate the enhanced membership inference attack
    """
    print("="*70)
    print("ENHANCED MEMBERSHIP INFERENCE ATTACK ON NEURAL NETWORK")
    print("="*70)
    
    # Run parameter sensitivity experiments
    results = run_parameter_experiments()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()