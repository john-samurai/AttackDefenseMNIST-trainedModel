import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# ==========================================
# TARGET MODEL (BANK'S Hand-Written Recognition SYSTEM) (mnist_cnn.pt)
# ==========================================
# This represents the bank's internal model
# The attacker CANNOT see this code.

class BankProprietaryModel(nn.Module):
    """
    This is the bank's secret model architecture.
    The attacker has NO ACCESS to this code.

    BankProprietaryModel is the "Container" (neural network structure)
    mnist_cnn.pt is the "Brain" (trained weights/knowledge)
    """
    def __init__(self):
        super(BankProprietaryModel, self).__init__()
        # Bank's secret architecture (same as original LeNet)
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

# ==========================================
# BANK'S API SERVICE
# ==========================================
# This simulates the bank's check processing API (Inference API).
# Attacker can only interact with this interface.

class BankCheckProcessingAPI:
    """
    This represents the bank's API service.
    The attacker can only call process_check_digit() method.
    They CANNOT see the internal model architecture.
    """
    
    def __init__(self, model_path="mnist_cnn.pt"):
        print("Bank's Check Processing API - Loading proprietary model...")
        
        # Load the bank's secret model
        # Step 1: Create empty container (architecture)
        self._secret_model = BankProprietaryModel()
        # Step 2: Load trained "brain" into container
        self._secret_model.load_state_dict(torch.load(model_path))
        # Step 3: Now it's a complete, working model
        self._secret_model.eval()
        
        print("Bank's proprietary model loaded successfully.")
        print("Model architecture is confidential and not accessible to clients.")
        
    def process_check_digit(self, digit_image):
        """
        PUBLIC API METHOD: Process handwritten digit from check
        This is the ONLY method attackers can access.
        
        Args:
            digit_image: Tensor of shape [batch_size, 1, 28, 28]
                batch_size: num of images, 1: channels (1 color means Grey scale),
                28: Height pixels, 28: Width pixels
        Returns:
            dict: API response with confidence scores
            (how sure the model is about each digit 0-9)
        """
        with torch.no_grad():
            # Internal processing (attacker can't see this)
            log_output = self._secret_model(digit_image)
            confidence_scores = torch.exp(log_output)
            
            # API response format (what attacker receives)
            api_response = {
                'status': 'success',
                'confidence_scores': confidence_scores.tolist(),
                'predicted_digit': confidence_scores.argmax(dim=1).tolist(),
                'max_confidence': confidence_scores.max(dim=1)[0].tolist()
            }
            
        return api_response

# ==========================================
# ATTACKER perspective, Creating a surrogate model
# ==========================================
# They have NO knowledge of the target architecture.

class AttackerSurrogateModel(nn.Module):
    """
    This is the attacker's guess at what architecture to use.
    They DON'T know the bank's real architecture.
    
    The attacker might try different architectures:
    - Simple CNN
        sequential neural network designed for image recognition.
    - ResNet-style
        solved the problem of training very deep neural networks.
    """
    def __init__(self, architecture_choice="simple_cnn"):
        super(AttackerSurrogateModel, self).__init__()
        
        if architecture_choice == "simple_cnn":
            # Attacker's guess: Simple CNN
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 7 * 7, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )
            
        elif architecture_choice == "resnet_style":
            # Attacker's guess: ResNet-inspired
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(64 * 7 * 7, 10)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture_choice}")
        
        self.architecture = architecture_choice
        
    def forward(self, x):
        if self.architecture == "simple_cnn":
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
            
        elif self.architecture == "resnet_style":
            x = F.relu(self.conv1(x))
            identity = x
            x = F.relu(self.conv2(x))
            x = x + identity  # Skip connection
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

class ModelExtractionAttacker:
    """
    This represents the attacker's complete system.
    They can only interact with the bank's API (no internal access).
    """
    
    def __init__(self):
        # Initialize connection to bank's API
        self.bank_api = BankCheckProcessingAPI()
        
        # Attacker's surrogate model (they choose the architecture)
        self.surrogate_model = None
        
        # Storage for stolen data
        self.stolen_inputs = []
        self.stolen_outputs = []
        
    def query_bank_api(self, images):
        """
        Attacker query using the inference API of the target model
        
        Args:
            images: Batch of images to process.
            e.g. batch_tensor = torch.stack(batch_images)  # Shape: [3, 1, 28, 28] (digits:3,7,0)
        Returns:
            confidence_scores: Only the API response.
            e.g.
            api_response = {
            'confidence_scores': [[0.01, 0.02, 0.01, 0.85, 0.02, 0.01, 0.01, 0.05, 0.01, 0.01],
                         [0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.89, 0.02, 0.01],
                         [0.91, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]]
                         }
        """
        # Call the bank's public API
        api_response = self.bank_api.process_check_digit(images)
        
        # Extract confidence scores from API response
        confidence_scores = torch.tensor(api_response['confidence_scores'])
        
        return confidence_scores
    
    def collect_training_data(self, num_samples=5000):
        """
        Phase 1-2: Collect input-output pairs by querying the API
        """
        print(f"\nATTACKER: Collecting {num_samples} input-output pairs")
        print("Using only black-box API access to bank's system")
        
        # Load diverse input samples
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        # Randomly sample for diversity
        indices = np.random.choice(len(train_dataset), num_samples, replace=False)
        
        # Query the API in batches
        batch_size = 64
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Prepare batch
            batch_images = []
            for idx in batch_indices:
                image, _ = train_dataset[idx]
                batch_images.append(image)
            
            batch_tensor = torch.stack(batch_images)
            
            # QUERY THE BANK'S API (This is the only thing attacker can do)
            api_response = self.query_bank_api(batch_tensor)
            
            # Store the stolen data
            self.stolen_inputs.append(batch_tensor)
            self.stolen_outputs.append(api_response)
            
            if i % (batch_size * 10) == 0:
                print(f"Progress: {i/len(indices)*100:.1f}%")
        
        # Combine all stolen data
        self.all_stolen_inputs = torch.cat(self.stolen_inputs, dim=0)
        self.all_stolen_outputs = torch.cat(self.stolen_outputs, dim=0)
        
        print(f"Collected {len(self.all_stolen_inputs)} input-output pairs")
        
    def train_surrogate_model(self, architecture_choice="simple_cnn"):
        """
        Phase 4: Train surrogate model with attacker's chosen architecture
        """
        print(f"\nATTACKER: Training surrogate model...")
        print(f"Chosen architecture: {architecture_choice}")
        print("Attacker doesn't know if this matches the bank's architecture.")
        
        # Create surrogate model with attacker's chosen architecture
        self.surrogate_model = AttackerSurrogateModel(architecture_choice)
        
        # Train using stolen data
        optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=0.001)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(self.all_stolen_inputs, self.all_stolen_outputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Training loop
        self.surrogate_model.train()
        for epoch in range(10):
            epoch_loss = 0
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = self.surrogate_model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(dataloader):.4f}")
        
        self.surrogate_model.eval()
        print("Surrogate model training completed.")
        
        # NEW: Save the attacker's trained surrogate model
        self.save_surrogate_model(architecture_choice)
        
    def save_surrogate_model(self, architecture_choice):
        """
        Save the attacker's trained surrogate model to a .pt file
        """
        # Create stolen_models directory if it doesn't exist
        if not os.path.exists('stolen_models'):
            os.makedirs('stolen_models')
            
        # Generate filename based on architecture and parameters
        num_samples = len(self.all_stolen_inputs)
        filename = f"stolen_models/attacker_surrogate_{architecture_choice}_{num_samples}samples.pt"
        
        # Save the model state dict
        torch.save(self.surrogate_model.state_dict(), filename)
        
        print(f"\nATTACKER'S MODEL SAVED.")
        print(f"Saved to: {filename}")
        print(f"Architecture: {architecture_choice}")
        print(f"Training samples used: {num_samples}")
        
        return filename
    
    def load_surrogate_model(self, model_path, architecture_choice):
        """
        Load a previously saved surrogate model
        """
        # Create model with specified architecture
        self.surrogate_model = AttackerSurrogateModel(architecture_choice)
        
        # Load trained weights
        self.surrogate_model.load_state_dict(torch.load(model_path))
        self.surrogate_model.eval()
        
        print(f"Loaded surrogate model from: {model_path}")
        print(f"Architecture: {architecture_choice}")
        
        return self.surrogate_model

def demonstrate_realistic_attack():
    """
    Demonstrate the realistic attack scenario
    """
    print("REALISTIC MODEL EXTRACTION ATTACK SIMULATION")
    print("=" * 60)
    print("Bank has proprietary check processing model")
    print("Attacker only has API access - no architecture knowledge.")
    print("=" * 60)
    
    # Initialize attacker
    attacker = ModelExtractionAttacker()
    
    # Show what attacker can and cannot do
    print(f"\nATTACKER CANNOT DO:")
    print("   - See bank's model architecture")
    print("   - Access model weights or parameters")
    print("   - View training data")
    print("   - Inspect internal computations")
    
    print(f"\nATTACKER CAN DO:")
    print("   - Send images to bank's Inference API")
    print("   - Receive confidence scores")
    print("   - Choose their own architecture for surrogate")
    print("   - Train surrogate using the bank's inference API responses")
    
    # Execute attack
    attacker.collect_training_data(num_samples=1000)
    attacker.train_surrogate_model(architecture_choice="simple_cnn")
    
    print(f"\nATTACK COMPLETED.")
    print("Attacker now has a working surrogate model")
    print("(without ever seeing the bank's architecture)")
    
    # Demonstrate loading the saved model
    print(f"\n" + "="*50)
    print("DEMONSTRATING MODEL PERSISTENCE:")
    print("="*50)
    
    # Create a new attacker instance to show loading works
    new_attacker = ModelExtractionAttacker()
    latest_model = "stolen_models/attacker_surrogate_simple_cnn_1000samples.pt"
    
    try:
        new_attacker.load_surrogate_model(latest_model, "simple_cnn")
        print("Successfully loaded previously trained surrogate model")
        print("Attacker's model has been created.")
    except FileNotFoundError:
        print("Model file not found - this is normal if running for the first time")

if __name__ == "__main__":
    demonstrate_realistic_attack()