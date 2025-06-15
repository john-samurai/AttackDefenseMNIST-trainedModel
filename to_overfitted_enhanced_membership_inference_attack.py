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

class OverfittedTargetModel(nn.Module):
    """
    Target model architecture (LeNet-style) that matches the overfitted model
    This simulates the bank's overfitted check processing model
    """
    def __init__(self):
        super(OverfittedTargetModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        # Note: No dropout layers in the overfitted model
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
        # No dropout during forward pass
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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
        # Slightly different architecture for shadow model
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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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

class OverfittedMembershipInferenceAttack:
    """
    Enhanced class implementing the membership inference attack specifically for overfitted models
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
        
        # Store training data indices used by the overfitted model
        self.target_training_indices = None
        
        # Timing statistics
        self.timing_stats = {
            'shadow_training_time': 0,
            'attack_training_time': 0,
            'inference_time': 0
        }
        
    def load_overfitted_target_model(self, model_path="mnist_cnn_overfitted.pt"):
        """Load the pre-trained overfitted target model"""
        print("Loading overfitted target model...")
        self.target_model = OverfittedTargetModel()
        try:
            self.target_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_model.eval()
            print("Overfitted target model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading overfitted target model: {e}")
            print("Please ensure mnist_cnn_overfitted.pt exists in the current directory")
            return False
    
    def get_overfitted_training_data(self, train_size_percentage=0.1):
        """
        Recreate the training data subset used by the overfitted model
        This simulates knowing which data was used for training (realistic in some scenarios)
        """
        print(f"Recreating overfitted model training data (using {train_size_percentage*100}% of MNIST)...")
        
        # Load full MNIST training dataset
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        # Recreate the same random split used by the overfitted model
        # Use the same random seed to get the exact same split
        torch.manual_seed(42)  # Same seed as used in a3_mnist_overfitted.py
        
        train_size = int(train_size_percentage * len(full_dataset))
        unused_size = len(full_dataset) - train_size
        
        # Get the same subset that was used for training the overfitted model
        train_subset, _ = torch.utils.data.random_split(full_dataset, [train_size, unused_size])
        
        # Extract the indices
        self.target_training_indices = set(train_subset.indices)
        
        print(f"Identified {len(self.target_training_indices)} samples used in overfitted model training")
        return train_subset
    
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
            # Use different random seeds for each shadow model
            torch.manual_seed(42 + i + 100)  # Different from target model seed
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
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
            
            # Initialize shadow model
            shadow_model = ShadowModel().to(self.device)
            optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
            
            # Train shadow model to be overfitted (similar to target)
            shadow_model.train()
            for epoch in range(15):  # More epochs to encourage overfitting
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = shadow_model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 5 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    print(f"  Epoch {epoch+1}/15, Average Loss: {avg_loss:.4f}")
            
            shadow_model.eval()
            self.shadow_models.append((shadow_model, train_data, test_data))
        
        self.timing_stats['shadow_training_time'] = time.time() - start_time
        print(f"Shadow models training completed in {self.timing_stats['shadow_training_time']:.2f} seconds!")
    
    def generate_attack_training_data(self, samples_per_class=300):
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
    
    def create_membership_test_set(self, num_members=500, num_non_members=500):
        """
        Create a test set with known membership labels for evaluation
        This uses knowledge of which samples were in the overfitted model's training set
        """
        print(f"\nCreating membership test set...")
        print(f"Members: {num_members}, Non-members: {num_non_members}")
        
        if self.target_training_indices is None:
            print("Error: Target training indices not available. Call get_overfitted_training_data() first.")
            return None, None
        
        # Load full MNIST training dataset
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        # Create member samples (from training set)
        member_indices = list(self.target_training_indices)[:num_members]
        member_samples = torch.utils.data.Subset(full_dataset, member_indices)
        member_labels = [1] * len(member_indices)  # 1 = member
        
        # Create non-member samples (from remaining data)
        all_indices = set(range(len(full_dataset)))
        non_member_indices = list(all_indices - self.target_training_indices)[:num_non_members]
        non_member_samples = torch.utils.data.Subset(full_dataset, non_member_indices)
        non_member_labels = [0] * len(non_member_indices)  # 0 = non-member
        
        # Combine samples and labels
        combined_samples = []
        combined_labels = []
        
        # Add member samples
        for i in range(len(member_samples)):
            data, _ = member_samples[i]
            combined_samples.append(data)
            combined_labels.append(1)
        
        # Add non-member samples
        for i in range(len(non_member_samples)):
            data, _ = non_member_samples[i]
            combined_samples.append(data)
            combined_labels.append(0)
        
        print(f"Created test set with {len(combined_samples)} samples")
        print(f"Actual members: {sum(combined_labels)}, Actual non-members: {len(combined_labels) - sum(combined_labels)}")
        
        return combined_samples, combined_labels
    
    def perform_membership_inference(self, test_samples, true_membership_labels, target_name="Overfitted Target Model"):
        """
        Step 4: Perform membership inference attack on overfitted target model
        """
        print(f"\nStep 4: Performing membership inference attack on {target_name}...")
        start_time = time.time()
        
        # Get confidence scores from target model
        predicted_memberships = []
        confidence_scores_list = []
        
        with torch.no_grad():
            for data in test_samples:
                data = data.unsqueeze(0).to(self.device)  # Add batch dimension
                
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
        
        # Analyze prediction probabilities
        member_probs = [predicted_memberships[i] for i in range(len(predicted_memberships)) 
                       if true_membership_labels[i] == 1]
        non_member_probs = [predicted_memberships[i] for i in range(len(predicted_memberships)) 
                           if true_membership_labels[i] == 0]
        
        print(f"\nPrediction Probability Analysis:")
        print(f"Member samples - Mean prob: {np.mean(member_probs):.4f}, Std: {np.std(member_probs):.4f}")
        print(f"Non-member samples - Mean prob: {np.mean(non_member_probs):.4f}, Std: {np.std(non_member_probs):.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': binary_predictions,
            'membership_probabilities': predicted_memberships,
            'confidence_scores': confidence_scores_list
        }
    
    def evaluate_overfitted_attack(self, num_shadow_models=3, samples_per_shadow_model=1000, 
                                  samples_per_class=300):
        """
        Comprehensive evaluation of the membership inference attack against overfitted model
        """
        print("\n" + "="*70)
        print("MEMBERSHIP INFERENCE ATTACK ON OVERFITTED MODEL")
        print("="*70)
        print(f"Configuration:")
        print(f"- Number of shadow models: {num_shadow_models}")
        print(f"- Samples per shadow model: {samples_per_shadow_model}")
        print(f"- Training samples per class: {samples_per_class}")
        print("="*70)
        
        # Step 1: Load overfitted target model
        if not self.load_overfitted_target_model():
            print("Failed to load overfitted target model. Exiting...")
            return None
        
        # Step 2: Get the training data used by overfitted model
        self.get_overfitted_training_data(train_size_percentage=0.1)
        
        # Step 3: Prepare shadow models
        shadow_datasets = self.prepare_shadow_data(num_shadow_models, samples_per_shadow_model)
        self.train_shadow_models(shadow_datasets)
        
        # Step 4: Train attack model
        attack_features, attack_labels = self.generate_attack_training_data(samples_per_class)
        self.train_attack_model(attack_features, attack_labels)
        
        # Step 5: Create membership test set
        test_samples, test_labels = self.create_membership_test_set(num_members=500, num_non_members=500)
        
        if test_samples is None:
            print("Failed to create test set. Exiting...")
            return None
        
        # Step 6: Perform inference attack
        results = self.perform_membership_inference(test_samples, test_labels)
        
        # Print timing summary
        print(f"\n=== TIMING ANALYSIS ===")
        print(f"Shadow Models Training: {self.timing_stats['shadow_training_time']:.2f} seconds")
        print(f"Attack Model Training: {self.timing_stats['attack_training_time']:.2f} seconds")
        print(f"Membership Inference: {self.timing_stats['inference_time']:.4f} seconds")
        print(f"Total Attack Time: {sum(self.timing_stats.values()):.2f} seconds")
        
        return results

def main():
    """
    Main function to demonstrate the membership inference attack on overfitted model
    """
    print("="*70)
    print("MEMBERSHIP INFERENCE ATTACK ON OVERFITTED NEURAL NETWORK")
    print("="*70)
    print("This attack targets the overfitted model created by a3_mnist_overfitted.py")
    print("The overfitting makes the model more vulnerable to membership inference.")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize attack
    attack = OverfittedMembershipInferenceAttack(device=device)
    
    # Run the attack evaluation
    results = attack.evaluate_overfitted_attack(
        num_shadow_models=3,
        samples_per_shadow_model=1000,
        samples_per_class=300
    )
    
    if results:
        print("\n" + "="*70)
        print("ATTACK COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Final Attack Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("The high accuracy demonstrates the vulnerability of overfitted models")
        print("to membership inference attacks.")
    else:
        print("Attack failed. Please check that mnist_cnn_overfitted.pt exists.")

if __name__ == "__main__":
    main()