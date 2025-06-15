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

class MinimalOverfitModel(nn.Module):
    """
    Target model architecture that matches your saved model
    This is the EXACT same architecture you used to create the overfitted model
    """
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

class ShadowModel(nn.Module):
    """
    Shadow model with similar but slightly different architecture
    Used to simulate target model behavior for generating attack training data
    """
    def __init__(self):
        super(ShadowModel, self).__init__()
        # Similar but different architecture for shadow model
        self.conv1 = nn.Conv2d(1, 6, 5, stride=2, padding=2)  # Different number of filters
        self.fc1 = nn.Linear(6 * 7 * 7, 40)  # Different size
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class ImprovedAttackModel(nn.Module):
    """
    Improved binary classifier to perform membership inference
    Uses multiple features from confidence scores
    """
    def __init__(self, input_size=13):  # Increased feature size
        super(ImprovedAttackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

class ImprovedMembershipInferenceAttack:
    """
    Improved membership inference attack with better feature engineering
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.target_model = None
        self.shadow_models = []
        self.attack_model = ImprovedAttackModel()
        
        # Use simpler transform to match your overfitted model (NO normalization)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Store training data indices used by the overfitted model
        self.target_training_indices = None
        
        # Timing statistics
        self.timing_stats = {
            'shadow_training_time': 0,
            'attack_training_time': 0,
            'inference_time': 0
        }
        
    def extract_enhanced_features(self, model_output, target_label=None):
        """
        Extract enhanced features from model output for better attack performance
        FIXED: Handle potential shape issues
        """
        # Convert log probabilities to probabilities
        probs = torch.exp(model_output)
        
        # Ensure we have the right shape [batch_size, num_classes]
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)  # Add batch dimension if missing
        
        # Handle single sample case
        if probs.shape[0] == 1:
            probs = probs.squeeze(0)  # Remove batch dimension for processing
        
        # Ensure we have exactly 10 classes (MNIST)
        if probs.shape[-1] != 10:
            print(f"Warning: Expected 10 classes, got {probs.shape[-1]}")
            # Pad or truncate to 10 classes
            if probs.shape[-1] < 10:
                padding = torch.zeros(10 - probs.shape[-1])
                probs = torch.cat([probs, padding])
            else:
                probs = probs[:10]
        
        # Basic confidence scores (10 features)
        confidence_scores = probs.cpu().numpy().flatten()
        
        # Additional features
        max_confidence = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        # Top-2 difference (difference between highest and second highest probability)
        # FIXED: Handle case where there might be fewer than 2 values
        if len(probs) >= 2:
            top2_values, _ = torch.topk(probs, 2)
            top2_diff = (top2_values[0] - top2_values[1]).item()
        else:
            # If only one class, set difference to the confidence score itself
            top2_diff = max_confidence
        
        # Combine all features
        enhanced_features = np.concatenate([
            confidence_scores,  # 10 features
            [max_confidence],   # 1 feature
            [entropy],          # 1 feature  
            [top2_diff]         # 1 feature
        ])
        
        return enhanced_features
        
    def load_overfitted_target_model(self, model_path="mnist_cnn_overfitted.pt"):
        """Load the pre-trained overfitted target model"""
        print("Loading overfitted target model...")
        self.target_model = MinimalOverfitModel()  # Use the CORRECT architecture
        try:
            self.target_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_model.eval()
            print("Overfitted target model loaded successfully!")
            print(f"Model architecture: {self.target_model}")
            
            # Test the model with a dummy input to verify it works
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 28, 28)
                output = self.target_model(dummy_input)
                print(f"Test output shape: {output.shape}")
                print(f"Test output (log probabilities): {output}")
                
            return True
        except Exception as e:
            print(f"Error loading overfitted target model: {e}")
            print("Please ensure mnist_cnn_overfitted.pt exists in the current directory")
            return False
    
    def get_overfitted_training_data(self, train_size=50):
        """
        Recreate the training data subset used by the overfitted model
        Updated to match your actual training setup (50 samples, not percentage)
        """
        print(f"Recreating overfitted model training data (using {train_size} samples)...")
        
        # Load full MNIST training dataset
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        # Recreate the EXACT same random split used by the overfitted model
        torch.manual_seed(42)
        unused_size = len(full_dataset) - train_size
        train_subset, _ = torch.utils.data.random_split(full_dataset, [train_size, unused_size])
        
        # Extract the indices
        self.target_training_indices = set(train_subset.indices)
        
        print(f"Identified {len(self.target_training_indices)} samples used in overfitted model training")
        return train_subset
    
    def prepare_shadow_data(self, num_models=5, samples_per_model=100):  # Smaller to match your setup
        """
        Prepare shadow datasets - use smaller datasets to encourage overfitting
        """
        print(f"\nStep 1: Preparing shadow datasets for {num_models} shadow models...")
        print(f"Each shadow model will use {samples_per_model} samples")
        
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        
        shadow_datasets = []
        
        for i in range(num_models):
            torch.manual_seed(42 + i + 200)
            indices = torch.randperm(len(full_dataset))[:samples_per_model]
            shadow_data = torch.utils.data.Subset(full_dataset, indices)
            shadow_datasets.append(shadow_data)
            print(f"Shadow dataset {i+1}: {len(shadow_data)} samples")
        
        return shadow_datasets
    
    def train_shadow_models(self, shadow_datasets):
        """
        Train shadow models to be highly overfitted
        """
        print(f"\nStep 2: Training {len(shadow_datasets)} shadow models...")
        start_time = time.time()
        
        self.shadow_models = []
        
        for i, shadow_data in enumerate(shadow_datasets):
            print(f"\nTraining Shadow Model {i+1}...")
            
            # Use very small training set to encourage overfitting
            train_size = int(0.5 * len(shadow_data))  # 50% for training
            test_size = len(shadow_data) - train_size
            train_data, test_data = torch.utils.data.random_split(shadow_data, [train_size, test_size])
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)  # Batch size 1
            
            # Initialize shadow model
            shadow_model = ShadowModel().to(self.device)
            optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)  # Start with Adam like your model
            
            # Train shadow model to be overfitted
            shadow_model.train()
            for epoch in range(100):  # More epochs
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = shadow_model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                if epoch % 20 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    
                    # Calculate training accuracy
                    shadow_model.eval()
                    train_correct = 0
                    with torch.no_grad():
                        for data, target in train_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = shadow_model(data)
                            pred = output.argmax(dim=1, keepdim=True)
                            train_correct += pred.eq(target.view_as(pred)).sum().item()
                    train_acc = 100. * train_correct / len(train_data)
                    
                    print(f"  Epoch {epoch+1}/100, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
                    shadow_model.train()
                    
                    # Early stopping if overfitting achieved
                    if train_acc > 90:
                        print(f"  Shadow model {i+1} achieved good overfitting, stopping early")
                        break
            
            shadow_model.eval()
            self.shadow_models.append((shadow_model, train_data, test_data))
        
        self.timing_stats['shadow_training_time'] = time.time() - start_time
        print(f"Shadow models training completed in {self.timing_stats['shadow_training_time']:.2f} seconds!")
    
    def generate_attack_training_data(self, samples_per_class=200):  # Reduced samples
        """
        Generate training data using enhanced features
        FIXED: Better error handling for model outputs
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
                    
                    # Debug: Print output shape for first few samples
                    if member_count < 3:
                        print(f"  Member sample {member_count}: output shape = {output.shape}")
                    
                    # Use enhanced features with better error handling
                    try:
                        enhanced_features = self.extract_enhanced_features(output)
                        attack_features.append(enhanced_features)
                        attack_labels.append(1)  # Member
                        member_count += 1
                    except Exception as e:
                        print(f"  Warning: Error processing member sample {member_count}: {e}")
                        continue
            
            # Generate non-member samples (from test data)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
            nonmember_count = 0
            
            with torch.no_grad():
                for data, _ in test_loader:
                    if nonmember_count >= samples_per_class:
                        break
                    
                    data = data.to(self.device)
                    output = shadow_model(data)
                    
                    # Debug: Print output shape for first few samples
                    if nonmember_count < 3:
                        print(f"  Non-member sample {nonmember_count}: output shape = {output.shape}")
                    
                    # Use enhanced features with better error handling
                    try:
                        enhanced_features = self.extract_enhanced_features(output)
                        attack_features.append(enhanced_features)
                        attack_labels.append(0)  # Non-member
                        nonmember_count += 1
                    except Exception as e:
                        print(f"  Warning: Error processing non-member sample {nonmember_count}: {e}")
                        continue
        
        if len(attack_features) == 0:
            raise ValueError("No attack features generated! Check shadow model outputs.")
        
        attack_features = np.array(attack_features)
        attack_labels = np.array(attack_labels)
        
        print(f"Generated {len(attack_features)} training samples for attack model")
        print(f"Feature dimension: {attack_features.shape[1]}")
        print(f"Members: {np.sum(attack_labels)}, Non-members: {len(attack_labels) - np.sum(attack_labels)}")
        
        return attack_features, attack_labels
    
    def train_attack_model(self, attack_features, attack_labels):
        """
        Train the improved attack model
        """
        print("\nStep 4: Training Attack Model...")
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize attack model
        self.attack_model = ImprovedAttackModel(input_size=attack_features.shape[1]).to(self.device)
        optimizer = optim.Adam(self.attack_model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Train attack model
        self.attack_model.train()
        best_val_acc = 0
        
        for epoch in range(100):  # More epochs
            epoch_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                predictions = self.attack_model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            if epoch % 10 == 0:
                self.attack_model.eval()
                with torch.no_grad():
                    val_predictions = self.attack_model(X_val)
                    val_loss = criterion(val_predictions, y_val)
                    val_accuracy = ((val_predictions > 0.5).float() == y_val).float().mean()
                    
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                    
                    print(f"Epoch {epoch+1}/100 - Train Loss: {epoch_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                self.attack_model.train()
        
        self.attack_model.eval()
        self.timing_stats['attack_training_time'] = time.time() - start_time
        print(f"Attack model training completed in {self.timing_stats['attack_training_time']:.2f} seconds!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    def create_membership_test_set(self, num_members=25, num_non_members=25):  # Smaller test set
        """
        Create a test set with known membership labels for evaluation
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
        member_samples = []
        for idx in member_indices:
            data, _ = full_dataset[idx]
            member_samples.append(data)
        
        # Create non-member samples (from remaining data)
        all_indices = set(range(len(full_dataset)))
        non_member_indices = list(all_indices - self.target_training_indices)[:num_non_members]
        non_member_samples = []
        for idx in non_member_indices:
            data, _ = full_dataset[idx]
            non_member_samples.append(data)
        
        # Combine samples and labels
        combined_samples = member_samples + non_member_samples
        combined_labels = [1] * len(member_samples) + [0] * len(non_member_samples)
        
        print(f"Created test set with {len(combined_samples)} samples")
        print(f"Actual members: {sum(combined_labels)}, Actual non-members: {len(combined_labels) - sum(combined_labels)}")
        
        return combined_samples, combined_labels
    
    def perform_membership_inference(self, test_samples, true_membership_labels, target_name="Overfitted Target Model"):
        """
        Perform membership inference attack using enhanced features
        """
        print(f"\nStep 5: Performing membership inference attack on {target_name}...")
        start_time = time.time()
        
        predicted_memberships = []
        confidence_scores_list = []
        
        with torch.no_grad():
            for i, data in enumerate(test_samples):
                data = data.unsqueeze(0).to(self.device)
                
                # Get target model output
                target_output = self.target_model(data)
                
                # Debug: Print shape for first few samples
                if i < 3:
                    print(f"  Test sample {i}: target output shape = {target_output.shape}")
                
                # Extract enhanced features
                try:
                    enhanced_features = self.extract_enhanced_features(target_output)
                    confidence_scores_list.append(enhanced_features[:10])  # Store original confidence scores
                    
                    # Use attack model to predict membership
                    attack_input = torch.FloatTensor(enhanced_features).unsqueeze(0).to(self.device)
                    membership_prob = self.attack_model(attack_input).item()
                    predicted_memberships.append(membership_prob)
                except Exception as e:
                    print(f"  Warning: Error processing test sample {i}: {e}")
                    # Use random prediction as fallback
                    predicted_memberships.append(0.5)
                    confidence_scores_list.append([0.1] * 10)
        
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
        
        if len(member_probs) > 0 and len(non_member_probs) > 0:
            print(f"\nPrediction Probability Analysis:")
            print(f"Member samples - Mean prob: {np.mean(member_probs):.4f}, Std: {np.std(member_probs):.4f}")
            print(f"Non-member samples - Mean prob: {np.mean(non_member_probs):.4f}, Std: {np.std(non_member_probs):.4f}")
            print(f"Probability difference: {np.mean(member_probs) - np.mean(non_member_probs):.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': binary_predictions,
            'membership_probabilities': predicted_memberships,
            'confidence_scores': confidence_scores_list
        }
    
    def evaluate_overfitted_attack(self, num_shadow_models=3, samples_per_shadow_model=100, 
                                  samples_per_class=200):
        """
        Comprehensive evaluation of the membership inference attack against overfitted model
        """
        print("\n" + "="*70)
        print("IMPROVED MEMBERSHIP INFERENCE ATTACK ON OVERFITTED MODEL")
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
        
        # Step 2: Get the training data used by overfitted model (50 samples)
        self.get_overfitted_training_data(train_size=50)
        
        # Step 3: Prepare shadow models
        shadow_datasets = self.prepare_shadow_data(num_shadow_models, samples_per_shadow_model)
        self.train_shadow_models(shadow_datasets)
        
        # Step 4: Train attack model
        attack_features, attack_labels = self.generate_attack_training_data(samples_per_class)
        self.train_attack_model(attack_features, attack_labels)
        
        # Step 5: Create membership test set
        test_samples, test_labels = self.create_membership_test_set(num_members=25, num_non_members=25)
        
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
    Main function to demonstrate the improved membership inference attack on overfitted model
    """
    print("="*70)
    print("IMPROVED MEMBERSHIP INFERENCE ATTACK ON OVERFITTED NEURAL NETWORK")
    print("="*70)
    print("This improved attack targets the severely overfitted model with:")
    print("- Enhanced feature engineering (13 features vs 10)")
    print("- Better shadow model training (more overfitted)")
    print("- Improved attack model architecture")
    print("- More aggressive overfitting parameters")
    print("- FIXED: Better error handling for model outputs")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize attack
    attack = ImprovedMembershipInferenceAttack(device=device)
    
    # Run the attack evaluation with smaller parameters to match your setup
    results = attack.evaluate_overfitted_attack(
        num_shadow_models=3,  # Fewer shadow models
        samples_per_shadow_model=100,  # Smaller shadow datasets
        samples_per_class=200  # Fewer training samples per class
    )
    
    if results:
        print("\n" + "="*70)
        print("ATTACK COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Final Attack Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        if results['accuracy'] > 0.7:
            print("EXCELLENT: High attack accuracy demonstrates severe overfitting vulnerability!")
        elif results['accuracy'] > 0.6:
            print("GOOD: Moderate attack accuracy shows overfitting vulnerability.")
        elif results['accuracy'] > 0.55:
            print("WEAK: Low attack accuracy suggests limited overfitting.")
        else:
            print("FAILED: Attack accuracy near random - model may not be overfitted enough.")
    else:
        print("Attack failed. Please check that mnist_cnn_overfitted.pt exists.")

if __name__ == "__main__":
    main()