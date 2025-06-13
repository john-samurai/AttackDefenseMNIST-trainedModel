import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ==========================================
# MODEL ARCHITECTURES (copied from main script)
# ==========================================

class BankProprietaryModel(nn.Module):
    """Target model architecture (LeNet)"""
    def __init__(self):
        super(BankProprietaryModel, self).__init__()
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

class AttackerSurrogateModel(nn.Module):
    """Attacker's surrogate model architecture"""
    def __init__(self, architecture_choice="simple_cnn"):
        super(AttackerSurrogateModel, self).__init__()
        
        if architecture_choice == "simple_cnn":
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
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(64 * 7 * 7, 10)
            
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
            x = x + identity
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

# ==========================================
# MODEL COMPARISON CLASS
# ==========================================

class ModelComparison:
    def __init__(self, target_model_path="mnist_cnn.pt", 
                 surrogate_model_path="stolen_models/attacker_surrogate_simple_cnn_1000samples.pt",
                 surrogate_architecture="simple_cnn"):
        
        print("MODEL PERFORMANCE EVALUATION")
        print("=" * 60)
        print("Comparing Target Model vs Attacker's Surrogate Model")
        print("=" * 60)
        
        # Load target model
        print("\nLoading Target Model (Bank's Proprietary)...")
        self.target_model = BankProprietaryModel()
        self.target_model.load_state_dict(torch.load(target_model_path))
        self.target_model.eval()
        print(f"Target model loaded from: {target_model_path}")
        
        # Load surrogate model
        print("\nLoading Surrogate Model (Attacker's Stolen)...")
        self.surrogate_model = AttackerSurrogateModel(surrogate_architecture)
        self.surrogate_model.load_state_dict(torch.load(surrogate_model_path))
        self.surrogate_model.eval()
        print(f"Surrogate model loaded from: {surrogate_model_path}")
        print(f"Architecture: {surrogate_architecture}")
        
        # Load test dataset
        self.load_test_data()
        
    def load_test_data(self):
        """Load MNIST test dataset for evaluation"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=100, shuffle=False
        )
        
        print(f"\nTest dataset loaded: {len(self.test_dataset)} samples")
    
    def get_model_predictions(self, model, num_samples=1000):
        """Get predictions from a model on test data"""
        predictions = []
        confidences = []
        true_labels = []
        
        model.eval()
        samples_collected = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                if samples_collected >= num_samples:
                    break
                    
                # Get model output
                output = model(data)
                confidence_scores = torch.exp(output)  # Convert log probabilities to probabilities
                
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                
                # Store results
                predictions.extend(pred.squeeze().tolist())
                confidences.extend(confidence_scores.tolist())
                true_labels.extend(target.tolist())
                
                samples_collected += data.size(0)
        
        # Trim to exact number of samples
        predictions = predictions[:num_samples]
        confidences = confidences[:num_samples]
        true_labels = true_labels[:num_samples]
        
        return np.array(predictions), np.array(confidences), np.array(true_labels)
    
    def calculate_agreement_rate(self, num_samples=1000):
        """Calculate how often both models make the same prediction"""
        print(f"\nCalculating Agreement Rate (using {num_samples} samples)...")
        
        # Get predictions from both models
        target_preds, target_conf, true_labels = self.get_model_predictions(self.target_model, num_samples)
        surrogate_preds, surrogate_conf, _ = self.get_model_predictions(self.surrogate_model, num_samples)
        
        # Calculate agreement
        agreements = (target_preds == surrogate_preds)
        agreement_rate = np.mean(agreements)
        
        print(f"Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
        print(f"Agreements: {np.sum(agreements)}/{num_samples}")
        
        # Calculate individual accuracies
        target_accuracy = np.mean(target_preds == true_labels)
        surrogate_accuracy = np.mean(surrogate_preds == true_labels)
        
        print(f"\nIndividual Model Accuracies:")
        print(f"Target Model Accuracy: {target_accuracy:.4f} ({target_accuracy*100:.2f}%)")
        print(f"Surrogate Model Accuracy: {surrogate_accuracy:.4f} ({surrogate_accuracy*100:.2f}%)")
        
        return agreement_rate, target_accuracy, surrogate_accuracy, target_preds, surrogate_preds, true_labels
    
    def calculate_confidence_similarity(self, num_samples=1000):
        """Calculate how similar the confidence scores are"""
        print(f"\nCalculating Confidence Score Similarity...")
        
        # Get confidence scores from both models
        _, target_conf, _ = self.get_model_predictions(self.target_model, num_samples)
        _, surrogate_conf, _ = self.get_model_predictions(self.surrogate_model, num_samples)
        
        # Calculate various similarity metrics
        # 1. KL Divergence (lower is better)
        kl_divergences = []
        for i in range(len(target_conf)):
            target_dist = np.array(target_conf[i]) + 1e-8  # Add small epsilon to avoid log(0)
            surrogate_dist = np.array(surrogate_conf[i]) + 1e-8
            
            # Normalize to ensure they're proper probability distributions
            target_dist = target_dist / np.sum(target_dist)
            surrogate_dist = surrogate_dist / np.sum(surrogate_dist)
            
            kl_div = np.sum(target_dist * np.log(target_dist / surrogate_dist))
            kl_divergences.append(kl_div)
        
        avg_kl_divergence = np.mean(kl_divergences)
        
        # 2. Cosine Similarity (higher is better)
        cosine_similarities = []
        for i in range(len(target_conf)):
            target_vec = np.array(target_conf[i])
            surrogate_vec = np.array(surrogate_conf[i])
            
            cosine_sim = np.dot(target_vec, surrogate_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(surrogate_vec))
            cosine_similarities.append(cosine_sim)
        
        avg_cosine_similarity = np.mean(cosine_similarities)
        
        print(f"Average KL Divergence: {avg_kl_divergence:.4f} (lower is better)")
        print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f} (higher is better)")
        
        return avg_kl_divergence, avg_cosine_similarity
    
    def analyze_disagreements(self, target_preds, surrogate_preds, true_labels, num_examples=10):
        """Analyze cases where models disagree"""
        print(f"\nAnalyzing Disagreements...")
        
        disagreements = (target_preds != surrogate_preds)
        disagreement_indices = np.where(disagreements)[0]
        
        print(f"Total disagreements: {np.sum(disagreements)}")
        print(f"Disagreement rate: {np.mean(disagreements)*100:.2f}%")
        
        if len(disagreement_indices) > 0:
            print(f"\nFirst {min(num_examples, len(disagreement_indices))} disagreement cases:")
            print("Index | True | Target | Surrogate | Target Correct | Surrogate Correct")
            print("-" * 70)
            
            for i, idx in enumerate(disagreement_indices[:num_examples]):
                true_label = true_labels[idx]
                target_pred = target_preds[idx]
                surrogate_pred = surrogate_preds[idx]
                target_correct = "YES" if target_pred == true_label else "NO"
                surrogate_correct = "YES" if surrogate_pred == true_label else "NO"
                
                print(f"{idx:5d} | {true_label:4d} | {target_pred:6d} | {surrogate_pred:9d} | {target_correct:14s} | {surrogate_correct}")
    
    def analyze_agreement_by_digit(self, target_preds, surrogate_preds, true_labels):
        """Analyze agreement rates for each digit (text output only)"""
        print(f"\nAgreement Rate Analysis by Digit:")
        print("=" * 50)
        
        agreements = (target_preds == surrogate_preds)
        
        print("Digit | Agreement Rate | Total Samples | Agreement Count")
        print("-" * 55)
        
        for digit in range(10):
            digit_mask = (true_labels == digit)
            if np.sum(digit_mask) > 0:
                digit_agreements = agreements[digit_mask]
                digit_agreement_rate = np.mean(digit_agreements)
                total_samples = np.sum(digit_mask)
                agreement_count = np.sum(digit_agreements)
                
                print(f"  {digit}   |     {digit_agreement_rate:.3f}      |      {total_samples:3d}      |      {agreement_count:3d}")
            else:
                print(f"  {digit}   |     N/A        |       0       |       0")
    
    def generate_confusion_matrix_text(self, predictions, true_labels, model_name):
        """Generate confusion matrix in text format"""
        print(f"\nConfusion Matrix for {model_name}:")
        print("=" * 40)
        
        # Create confusion matrix
        cm = np.zeros((10, 10), dtype=int)
        for true_label, pred_label in zip(true_labels, predictions):
            cm[true_label][pred_label] += 1
        
        # Print header
        print("True\\Pred", end="")
        for j in range(10):
            print(f"{j:4d}", end="")
        print()
        
        # Print matrix
        for i in range(10):
            print(f"   {i:1d}    ", end="")
            for j in range(10):
                print(f"{cm[i][j]:4d}", end="")
            print()
        
        # Calculate per-class accuracy
        print(f"\nPer-class accuracy for {model_name}:")
        for i in range(10):
            if np.sum(cm[i, :]) > 0:
                accuracy = cm[i, i] / np.sum(cm[i, :])
                print(f"Digit {i}: {accuracy:.3f} ({cm[i, i]}/{np.sum(cm[i, :])})")
            else:
                print(f"Digit {i}: N/A (no samples)")
    
    def generate_comprehensive_report(self, num_samples=1000):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        # Calculate main metrics
        agreement_rate, target_acc, surrogate_acc, target_preds, surrogate_preds, true_labels = self.calculate_agreement_rate(num_samples)
        
        # Calculate confidence similarities
        kl_div, cosine_sim = self.calculate_confidence_similarity(num_samples)
        
        # Analyze disagreements
        self.analyze_disagreements(target_preds, surrogate_preds, true_labels)
        
        # Analyze agreement by digit
        self.analyze_agreement_by_digit(target_preds, surrogate_preds, true_labels)
        
        # Generate confusion matrices in text format
        self.generate_confusion_matrix_text(target_preds, true_labels, "Target Model")
        self.generate_confusion_matrix_text(surrogate_preds, true_labels, "Surrogate Model")
        
        # Summary statistics
        print(f"\n" + "="*50)
        print("ATTACK SUCCESS SUMMARY")
        print("="*50)
        print(f"Model Agreement Rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")
        print(f"Target Model Accuracy: {target_acc:.4f} ({target_acc*100:.2f}%)")
        print(f"Surrogate Model Accuracy: {surrogate_acc:.4f} ({surrogate_acc*100:.2f}%)")
        print(f"Confidence Similarity (Cosine): {cosine_sim:.4f}")
        print(f"Confidence Divergence (KL): {kl_div:.4f}")
        
        # Attack success assessment
        print(f"\nATTACK SUCCESS ASSESSMENT:")
        if agreement_rate > 0.95:
            print("EXCELLENT: Surrogate model almost perfectly mimics target!")
        elif agreement_rate > 0.90:
            print("VERY GOOD: High functional similarity achieved!")
        elif agreement_rate > 0.85:
            print("GOOD: Substantial functional replication!")
        elif agreement_rate > 0.80:
            print("MODERATE: Partial success, room for improvement!")
        else:
            print("POOR: Attack needs refinement!")
        
        # For assignment report
        print(f"\n" + "="*50)
        print("FOR YOUR ASSIGNMENT REPORT:")
        print("="*50)
        print(f"Agreement Rate: {agreement_rate*100:.1f}%")
        print(f"Target Model Accuracy: {target_acc*100:.1f}%")
        print(f"Surrogate Model Accuracy: {surrogate_acc*100:.1f}%")
        print(f"Confidence Similarity: {cosine_sim:.3f}")
        print(f"KL Divergence: {kl_div:.3f}")
        print(f"Samples Used for Training: 1000 API queries")
        print(f"Attack Success Level: {'EXCELLENT' if agreement_rate > 0.95 else 'VERY GOOD' if agreement_rate > 0.90 else 'GOOD' if agreement_rate > 0.85 else 'MODERATE' if agreement_rate > 0.80 else 'POOR'}")
        
        return {
            'agreement_rate': agreement_rate,
            'target_accuracy': target_acc,
            'surrogate_accuracy': surrogate_acc,
            'kl_divergence': kl_div,
            'cosine_similarity': cosine_sim
        }

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_model_comparison():
    """Run the complete model comparison analysis"""
    try:
        # Initialize comparison
        comparison = ModelComparison(
            target_model_path="mnist_cnn.pt",
            surrogate_model_path="stolen_models/attacker_surrogate_simple_cnn_1000samples.pt",
            surrogate_architecture="simple_cnn"
        )
        
        # Generate comprehensive report
        results = comparison.generate_comprehensive_report(num_samples=1000)
        
        print(f"\nComparison complete!")
        print("All results displayed above - no image files generated.")
        
        return results
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have run the attack script first to generate the surrogate model!")
        print("Required files:")
        print("- mnist_cnn.pt (target model)")
        print("- stolen_models/attacker_surrogate_simple_cnn_1000samples.pt (surrogate model)")

if __name__ == "__main__":
    run_model_comparison()