import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import hashlib
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import from the existing attack file
try:
    from demonstrate_realistic_attack import (
        BankProprietaryModel, 
        BankCheckProcessingAPI,
        AttackerSurrogateModel,
        ModelExtractionAttacker
    )
    print("Successfully imported from demonstrate_realistic_attack.py")
except ImportError as e:
    print(f"Warning: Could not import from demonstrate_realistic_attack.py: {e}")
    print("Please ensure demonstrate_realistic_attack.py is in the same directory")

# ==========================================
# STEP 1: QUERY PATTERN DETECTION
# ==========================================
class QueryPatternDetector:
    """Detects suspicious query patterns that indicate model extraction attempts"""
    
    def __init__(self, window_size=50, diversity_threshold=0.6, rate_threshold=30):
        self.window_size = window_size
        self.diversity_threshold = diversity_threshold
        self.rate_threshold = rate_threshold
        self.query_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.user_patterns = defaultdict(lambda: {'queries': deque(maxlen=window_size), 
                                                 'timestamps': deque(maxlen=window_size)})
        # Timing tracking for Step 1
        self.timing_stats = {
            'total_detection_time': 0.0,
            'detection_count': 0,
            'min_detection_time': float('inf'),
            'max_detection_time': 0.0,
            'avg_detection_time': 0.0
        }
    
    def _calculate_diversity(self, images):
        """Calculate diversity of images in the batch using pixel variance"""
        if len(images) < 2:
            return 0.0
        
        # Flatten images and calculate pairwise cosine similarities
        flattened = images.view(images.size(0), -1).numpy()
        similarities = cosine_similarity(flattened)
        
        # High diversity = low average similarity
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        diversity = 1.0 - avg_similarity
        return diversity
    
    def _calculate_query_rate(self, user_id):
        """Calculate recent query rate for a user"""
        if len(self.user_patterns[user_id]['timestamps']) < 2:
            return 0.0
        
        recent_times = list(self.user_patterns[user_id]['timestamps'])
        time_span = recent_times[-1] - recent_times[0]
        
        if time_span == 0:
            return float('inf')  # All queries at same time - very suspicious
        
        return len(recent_times) / time_span
    
    def detect_suspicious_pattern(self, images, user_id="default"):
        """
        Detect if the current query shows suspicious patterns
        Returns: (is_suspicious, suspicion_details)
        """
        # START TIMING FOR STEP 1
        detection_start = time.perf_counter()
        
        current_time = time.time()
        suspicion_details = {}
        
        # Calculate image diversity
        diversity = self._calculate_diversity(images)
        suspicion_details['diversity'] = diversity
        
        # Update user history
        self.user_patterns[user_id]['queries'].append(images)
        self.user_patterns[user_id]['timestamps'].append(current_time)
        
        # Calculate query rate
        query_rate = self._calculate_query_rate(user_id)
        suspicion_details['query_rate'] = query_rate
        
        # Detection logic
        is_suspicious = False
        reasons = []
        
        # High diversity indicates random sampling (typical in model extraction)
        if diversity > self.diversity_threshold:
            is_suspicious = True
            reasons.append(f"High image diversity: {diversity:.3f}")
        
        # High query rate indicates automated querying
        if query_rate > self.rate_threshold:
            is_suspicious = True
            reasons.append(f"High query rate: {query_rate:.2f} queries/second")
        
        # Batch size patterns (model extraction often uses specific batch sizes)
        batch_size = images.size(0)
        if batch_size >= 32:
            suspicion_details['suspicious_batch_size'] = batch_size
            reasons.append(f"Suspicious batch size: {batch_size}")
            is_suspicious = True
        
        # Sequential pattern detection - attackers often query many random samples rapidly
        if len(self.user_patterns[user_id]['queries']) >= 5:
            recent_diversities = []
            for recent_query in list(self.user_patterns[user_id]['queries'])[-5:]:
                if recent_query.size(0) > 1:  # Only calculate for batches
                    recent_div = self._calculate_diversity(recent_query)
                    recent_diversities.append(recent_div)
            
            if len(recent_diversities) >= 3:
                avg_recent_diversity = np.mean(recent_diversities)
                if avg_recent_diversity > 0.5:  # Consistently high diversity
                    is_suspicious = True
                    reasons.append(f"Consistent high diversity pattern: {avg_recent_diversity:.3f}")
                    suspicion_details['pattern_diversity'] = avg_recent_diversity
        
        suspicion_details['reasons'] = reasons
        suspicion_details['is_suspicious'] = is_suspicious
        
        # END TIMING FOR STEP 1
        detection_end = time.perf_counter()
        detection_time = detection_end - detection_start
        
        # Update timing statistics
        self.timing_stats['total_detection_time'] += detection_time
        self.timing_stats['detection_count'] += 1
        self.timing_stats['min_detection_time'] = min(self.timing_stats['min_detection_time'], detection_time)
        self.timing_stats['max_detection_time'] = max(self.timing_stats['max_detection_time'], detection_time)
        self.timing_stats['avg_detection_time'] = self.timing_stats['total_detection_time'] / self.timing_stats['detection_count']
        
        return is_suspicious, suspicion_details

# ==========================================
# STEP 2: ADAPTIVE CONFIDENCE SCORE DEVIATION
# ==========================================
class ConfidenceScoreDefender:
    """Applies adaptive noise to confidence scores based on suspicion level"""
    
    def __init__(self, base_noise=0.2, max_noise=0.8):
        self.base_noise = base_noise
        self.max_noise = max_noise
        # Timing tracking for Step 2
        self.timing_stats = {
            'total_noise_time': 0.0,
            'noise_count': 0,
            'min_noise_time': float('inf'),
            'max_noise_time': 0.0,
            'avg_noise_time': 0.0
        }
    
    def add_defensive_noise(self, confidence_scores, is_suspicious, suspicion_details):
        """
        Add calibrated noise to confidence scores
        - Preserves predicted digit for legitimate users
        - Corrupts confidence patterns for attackers
        """
        # START TIMING FOR STEP 2
        noise_start = time.perf_counter()
        
        if not is_suspicious:
            # Minimal noise for legitimate users
            noise_level = self.base_noise * 0.1
        else:
            # Scale noise based on suspicion level
            suspicion_factors = []
            
            if suspicion_details.get('diversity', 0) > 0.6:
                suspicion_factors.append(suspicion_details['diversity'])
            
            if suspicion_details.get('query_rate', 0) > 30:
                suspicion_factors.append(min(suspicion_details['query_rate'] / 60, 1.0))
            
            if suspicion_details.get('suspicious_batch_size'):
                suspicion_factors.append(0.3)
            
            avg_suspicion = np.mean(suspicion_factors) if suspicion_factors else 0.5
            noise_level = self.base_noise + (self.max_noise - self.base_noise) * avg_suspicion
        
        # Apply noise while preserving the predicted class
        original_predictions = confidence_scores.argmax(dim=1)
        
        # Add Gaussian noise to confidence scores
        noise = torch.normal(0, noise_level, size=confidence_scores.shape)
        noisy_scores = confidence_scores + noise
        
        # Ensure probabilities remain valid (positive and sum to 1)
        noisy_scores = torch.softmax(noisy_scores, dim=1)
        
        # Verification: ensure predicted class didn't change for legitimate users
        new_predictions = noisy_scores.argmax(dim=1)
        if not is_suspicious:
            # For legitimate users, restore original prediction if it changed
            changed_mask = (original_predictions != new_predictions)
            if changed_mask.any():
                for i in range(len(noisy_scores)):
                    if changed_mask[i]:
                        # Boost the original prediction
                        original_class = original_predictions[i]
                        noisy_scores[i][original_class] = noisy_scores[i].max() + 0.1
                        noisy_scores[i] = torch.softmax(noisy_scores[i], dim=0)
        
        # END TIMING FOR STEP 2
        noise_end = time.perf_counter()
        noise_time = noise_end - noise_start
        
        # Update timing statistics
        self.timing_stats['total_noise_time'] += noise_time
        self.timing_stats['noise_count'] += 1
        self.timing_stats['min_noise_time'] = min(self.timing_stats['min_noise_time'], noise_time)
        self.timing_stats['max_noise_time'] = max(self.timing_stats['max_noise_time'], noise_time)
        self.timing_stats['avg_noise_time'] = self.timing_stats['total_noise_time'] / self.timing_stats['noise_count']
        
        return noisy_scores, noise_level

# ==========================================
# STEP 3: RATE LIMITING WITH BEHAVIORAL ANALYSIS
# ==========================================
class BehavioralRateLimiter:
    """Dynamic rate limiting based on user behavior patterns"""
    
    def __init__(self, base_limit=50, suspicious_limit=5, time_window=3600):
        self.base_limit = base_limit  # queries per hour for legitimate users
        self.suspicious_limit = suspicious_limit  # queries per hour for suspicious users
        self.time_window = time_window  # 1 hour in seconds
        self.user_history = defaultdict(lambda: {'queries': deque(), 'suspicion_score': 0.0})
        # Timing tracking for Step 3
        self.timing_stats = {
            'total_limiting_time': 0.0,
            'limiting_count': 0,
            'min_limiting_time': float('inf'),
            'max_limiting_time': 0.0,
            'avg_limiting_time': 0.0
        }
    
    def update_suspicion_score(self, user_id, is_suspicious, suspicion_details):
        """Update user's suspicion score based on recent behavior"""
        current_score = self.user_history[user_id]['suspicion_score']
        
        if is_suspicious:
            # Increase suspicion more aggressively
            score_increase = len(suspicion_details.get('reasons', [])) * 0.3
            current_score = min(1.0, current_score + score_increase)
        else:
            # Decrease suspicion over time (rehabilitation) - slower recovery
            current_score = max(0.0, current_score - 0.02)
        
        self.user_history[user_id]['suspicion_score'] = current_score
    
    def check_rate_limit(self, user_id, is_suspicious, suspicion_details):
        """
        Check if user has exceeded their rate limit
        Returns: (is_allowed, wait_time_seconds)
        """
        # START TIMING FOR STEP 3
        limiting_start = time.perf_counter()
        
        current_time = time.time()
        user_data = self.user_history[user_id]
        
        # Update suspicion score
        self.update_suspicion_score(user_id, is_suspicious, suspicion_details)
        
        # Clean old queries outside time window
        while user_data['queries'] and current_time - user_data['queries'][0] > self.time_window:
            user_data['queries'].popleft()
        
        # Determine rate limit based on suspicion
        suspicion_score = user_data['suspicion_score']
        current_limit = self.base_limit * (1 - suspicion_score) + self.suspicious_limit * suspicion_score
        
        # Check if under limit
        current_queries = len(user_data['queries'])
        
        if current_queries < current_limit:
            user_data['queries'].append(current_time)
            is_allowed = True
            wait_time = 0.0
        else:
            # Calculate wait time until oldest query expires
            oldest_query_time = user_data['queries'][0]
            wait_time = oldest_query_time + self.time_window - current_time
            is_allowed = False
            wait_time = max(0, wait_time)
        
        # END TIMING FOR STEP 3
        limiting_end = time.perf_counter()
        limiting_time = limiting_end - limiting_start
        
        # Update timing statistics
        self.timing_stats['total_limiting_time'] += limiting_time
        self.timing_stats['limiting_count'] += 1
        self.timing_stats['min_limiting_time'] = min(self.timing_stats['min_limiting_time'], limiting_time)
        self.timing_stats['max_limiting_time'] = max(self.timing_stats['max_limiting_time'], limiting_time)
        self.timing_stats['avg_limiting_time'] = self.timing_stats['total_limiting_time'] / self.timing_stats['limiting_count']
        
        return is_allowed, wait_time

# ==========================================
# INTEGRATED DEFENDED API
# ==========================================
class DefendedBankCheckProcessingAPI:
    """Enhanced bank API with comprehensive defense mechanisms"""
    
    def __init__(self, model_path="mnist_cnn.pt"):
        print("Initializing Defended Bank's Check Processing API...")
        
        # Load original model
        self._secret_model = BankProprietaryModel()
        self._secret_model.load_state_dict(torch.load(model_path))
        self._secret_model.eval()
        
        # Initialize defense components
        self.pattern_detector = QueryPatternDetector()
        self.confidence_defender = ConfidenceScoreDefender()
        self.rate_limiter = BehavioralRateLimiter()
        
        # Defense statistics
        self.defense_stats = {
            'total_queries': 0,
            'blocked_queries': 0,
            'suspicious_queries': 0,
            'legitimate_queries': 0,
            'noise_levels': []
        }
        
        # Overall timing tracking
        self.overall_timing = {
            'total_processing_time': 0.0,
            'original_model_time': 0.0,
            'query_count': 0,
            'avg_processing_time': 0.0,
            'avg_original_model_time': 0.0
        }
        
        print("Defense mechanisms activated:")
        print("- Query Pattern Detection")
        print("- Adaptive Confidence Score Deviation") 
        print("- Behavioral Rate Limiting")
    
    def process_check_digit(self, digit_image, user_id="default"):
        """
        Process digit recognition request with comprehensive defense
        """
        # START OVERALL TIMING
        overall_start = time.perf_counter()
        
        self.defense_stats['total_queries'] += 1
        
        # STEP 1: Query Pattern Detection
        is_suspicious, suspicion_details = self.pattern_detector.detect_suspicious_pattern(
            digit_image, user_id)
        
        if is_suspicious:
            self.defense_stats['suspicious_queries'] += 1
            print(f"SUSPICIOUS QUERY DETECTED from {user_id}")
            print(f"Reasons: {', '.join(suspicion_details['reasons'])}")
        else:
            self.defense_stats['legitimate_queries'] += 1
        
        # STEP 3: Rate Limiting Check
        is_allowed, wait_time = self.rate_limiter.check_rate_limit(
            user_id, is_suspicious, suspicion_details)
        
        if not is_allowed:
            self.defense_stats['blocked_queries'] += 1
            
            # END TIMING FOR BLOCKED QUERIES
            overall_end = time.perf_counter()
            overall_time = overall_end - overall_start
            self.overall_timing['total_processing_time'] += overall_time
            self.overall_timing['query_count'] += 1
            self.overall_timing['avg_processing_time'] = self.overall_timing['total_processing_time'] / self.overall_timing['query_count']
            
            return {
                'status': 'rate_limited',
                'message': f'Rate limit exceeded. Try again in {wait_time:.1f} seconds',
                'wait_time': wait_time
            }
        
        # Process request with original model (TIME THIS SEPARATELY)
        model_start = time.perf_counter()
        with torch.no_grad():
            log_output = self._secret_model(digit_image)
            confidence_scores = torch.exp(log_output)
        model_end = time.perf_counter()
        model_time = model_end - model_start
        
        # STEP 2: Apply Defensive Noise to Confidence Scores
        defended_scores, noise_level = self.confidence_defender.add_defensive_noise(
            confidence_scores, is_suspicious, suspicion_details)
        
        self.defense_stats['noise_levels'].append(noise_level)
        
        # Prepare API response
        api_response = {
            'status': 'success',
            'confidence_scores': defended_scores.tolist(),
            'predicted_digit': defended_scores.argmax(dim=1).tolist(),
            'max_confidence': defended_scores.max(dim=1)[0].tolist(),
            # Defense metadata (normally hidden from attackers)
            'defense_info': {
                'suspicious': is_suspicious,
                'noise_level': noise_level,
                'user_suspicion_score': self.rate_limiter.user_history[user_id]['suspicion_score']
            }
        }
        
        # END OVERALL TIMING
        overall_end = time.perf_counter()
        overall_time = overall_end - overall_start
        
        # Update timing statistics
        self.overall_timing['total_processing_time'] += overall_time
        self.overall_timing['original_model_time'] += model_time
        self.overall_timing['query_count'] += 1
        self.overall_timing['avg_processing_time'] = self.overall_timing['total_processing_time'] / self.overall_timing['query_count']
        self.overall_timing['avg_original_model_time'] = self.overall_timing['original_model_time'] / self.overall_timing['query_count']
        
        return api_response
    
    def get_defense_statistics(self):
        """Return comprehensive defense statistics"""
        total = self.defense_stats['total_queries']
        if total == 0:
            return "No queries processed yet"
        
        avg_noise = np.mean(self.defense_stats['noise_levels']) if self.defense_stats['noise_levels'] else 0
        
        stats = f"""
=== DEFENSE STATISTICS ===
Total Queries: {total}
Legitimate Queries: {self.defense_stats['legitimate_queries']} ({100*self.defense_stats['legitimate_queries']/total:.1f}%)
Suspicious Queries: {self.defense_stats['suspicious_queries']} ({100*self.defense_stats['suspicious_queries']/total:.1f}%)
Blocked Queries: {self.defense_stats['blocked_queries']} ({100*self.defense_stats['blocked_queries']/total:.1f}%)
Average Noise Level: {avg_noise:.3f}
"""
        return stats
    
    def get_timing_analysis(self):
        """Return comprehensive timing analysis"""
        if self.overall_timing['query_count'] == 0:
            return "No timing data available"
        
        # Calculate defense overhead
        defense_overhead = self.overall_timing['avg_processing_time'] - self.overall_timing['avg_original_model_time']
        overhead_percentage = (defense_overhead / self.overall_timing['avg_original_model_time']) * 100 if self.overall_timing['avg_original_model_time'] > 0 else 0
        
        timing_report = f"""
=== TIMING ANALYSIS ===
Query Processing Performance:
  Total Queries Processed: {self.overall_timing['query_count']}
  Average Processing Time: {self.overall_timing['avg_processing_time']*1000:.4f} ms
  Average Original Model Time: {self.overall_timing['avg_original_model_time']*1000:.4f} ms
  Defense Overhead: {defense_overhead*1000:.4f} ms ({overhead_percentage:.2f}%)

Step 1 - Query Pattern Detection:
  Average Detection Time: {self.pattern_detector.timing_stats['avg_detection_time']*1000:.4f} ms
  Min Detection Time: {self.pattern_detector.timing_stats['min_detection_time']*1000:.4f} ms
  Max Detection Time: {self.pattern_detector.timing_stats['max_detection_time']*1000:.4f} ms
  Total Detections: {self.pattern_detector.timing_stats['detection_count']}

Step 2 - Confidence Score Defense:
  Average Noise Application Time: {self.confidence_defender.timing_stats['avg_noise_time']*1000:.4f} ms
  Min Noise Time: {self.confidence_defender.timing_stats['min_noise_time']*1000:.4f} ms
  Max Noise Time: {self.confidence_defender.timing_stats['max_noise_time']*1000:.4f} ms
  Total Noise Applications: {self.confidence_defender.timing_stats['noise_count']}

Step 3 - Rate Limiting:
  Average Rate Check Time: {self.rate_limiter.timing_stats['avg_limiting_time']*1000:.4f} ms
  Min Rate Check Time: {self.rate_limiter.timing_stats['min_limiting_time']*1000:.4f} ms
  Max Rate Check Time: {self.rate_limiter.timing_stats['max_limiting_time']*1000:.4f} ms
  Total Rate Checks: {self.rate_limiter.timing_stats['limiting_count']}

Performance Summary:
  Real-time Processing: {'YES' if self.overall_timing['avg_processing_time'] < 0.1 else 'NO'}
  Negligible Latency: {'YES' if defense_overhead < 0.01 else 'NO'}
  Efficiency Rating: {'EXCELLENT' if overhead_percentage < 10 else 'GOOD' if overhead_percentage < 25 else 'ACCEPTABLE'}
"""
        return timing_report

# ==========================================
# DEFENDED ATTACKER CLASS (Modified from original)
# ==========================================
class DefendedModelExtractionAttacker:
    """Modified attacker that works with defended API"""
    
    def __init__(self, defended_api):
        # Use defended API instead of original
        self.bank_api = defended_api
        
        # Attacker's surrogate model (they choose the architecture)
        self.surrogate_model = None
        
        # Storage for stolen data
        self.stolen_inputs = []
        self.stolen_outputs = []
        
        # Track defense impact
        self.blocked_queries = 0
        self.successful_queries = 0
        
        # Timer tracking
        self.timing_results = {}
        
    def query_bank_api(self, images, user_id="attacker"):
        """
        Modified to handle defended API responses
        """
        # Call the defended bank's API
        api_response = self.bank_api.process_check_digit(images, user_id)
        
        # Check if query was blocked
        if api_response['status'] == 'rate_limited':
            self.blocked_queries += len(images)
            print(f"Query blocked - Rate limited. Wait time: {api_response.get('wait_time', 0):.1f}s")
            return None
        
        # Extract confidence scores from successful API response
        confidence_scores = torch.tensor(api_response['confidence_scores'])
        self.successful_queries += len(images)
        
        return confidence_scores
    
    def collect_training_data(self, num_samples=1000, user_id="attacker"):
        """
        Modified data collection that handles defense mechanisms
        """
        print(f"\nATTACKER: Attempting to collect {num_samples} input-output pairs")
        print("Using black-box API access to defended bank system")
        
        # Start timing data collection
        data_collection_start = time.time()
        
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
        
        # Query the API in smaller batches to avoid detection
        batch_size = 16  # Smaller batch size to try to evade detection
        collected_samples = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Prepare batch
            batch_images = []
            for idx in batch_indices:
                image, _ = train_dataset[idx]
                batch_images.append(image)
            
            batch_tensor = torch.stack(batch_images)
            
            # QUERY THE DEFENDED BANK'S API
            api_response = self.query_bank_api(batch_tensor, user_id)
            
            if api_response is not None:
                # Store the stolen data
                self.stolen_inputs.append(batch_tensor)
                self.stolen_outputs.append(api_response)
                collected_samples += len(batch_indices)
                
                if i % (batch_size * 10) == 0:
                    print(f"Progress: {collected_samples}/{num_samples} samples collected")
            else:
                # Query was blocked, attacker might stop or wait
                print(f"Attack hindered at {collected_samples} samples")
                break
        
        # Combine all stolen data
        if self.stolen_inputs:
            self.all_stolen_inputs = torch.cat(self.stolen_inputs, dim=0)
            self.all_stolen_outputs = torch.cat(self.stolen_outputs, dim=0)
        else:
            self.all_stolen_inputs = torch.empty(0, 1, 28, 28)
            self.all_stolen_outputs = torch.empty(0, 10)
        
        # End timing data collection
        data_collection_end = time.time()
        self.timing_results['data_collection_time'] = data_collection_end - data_collection_start
        
        print(f"Attack Results:")
        print(f"- Collected {len(self.all_stolen_inputs)} samples (target: {num_samples})")
        print(f"- Successful queries: {self.successful_queries}")
        print(f"- Blocked queries: {self.blocked_queries}")
        print(f"- Success rate: {self.successful_queries/(self.successful_queries + self.blocked_queries)*100:.1f}%")
        print(f"- Data collection time: {self.timing_results['data_collection_time']:.2f} seconds")
        
    def train_surrogate_model(self, architecture_choice="simple_cnn"):
        """
        Modified training that handles potentially limited/noisy data
        """
        if len(self.all_stolen_inputs) == 0:
            print("No training data collected - cannot train surrogate model")
            return
        
        print(f"\nATTACKER: Training surrogate model with {len(self.all_stolen_inputs)} samples...")
        print(f"Chosen architecture: {architecture_choice}")
        
        # Start timing surrogate model creation and training
        surrogate_start_time = time.time()
        
        # Create surrogate model with attacker's chosen architecture
        self.surrogate_model = AttackerSurrogateModel(architecture_choice)
        
        # Train using stolen data (which may be noisy due to defense)
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
        
        # End timing
        surrogate_end_time = time.time()
        self.timing_results['total_surrogate_time'] = surrogate_end_time - surrogate_start_time
        
        self.surrogate_model.eval()
        print("Surrogate model training completed.")
        print(f"Total surrogate model process time: {self.timing_results['total_surrogate_time']:.2f} seconds")

# ==========================================
# DEFENSE EVALUATION FRAMEWORK
# ==========================================
def evaluate_defense_effectiveness():
    """Comprehensive evaluation of defense vs original attack"""
    
    print("="*70)
    print("DEFENSE EFFECTIVENESS EVALUATION")
    print("="*70)
    
    # Initialize original and defended APIs
    print("\nInitializing systems...")
    original_api = BankCheckProcessingAPI()  # From demonstrate_realistic_attack.py
    defended_api = DefendedBankCheckProcessingAPI()
    
    # Test 1: Simulate legitimate users
    print("\n" + "="*50)
    print("TEST 1: LEGITIMATE USER EXPERIENCE")
    print("="*50)
    
    legitimate_success_rate, legitimate_accuracy = test_legitimate_users(defended_api)
    
    # Test 2: Compare original attack vs defended attack
    print("\n" + "="*50)
    print("TEST 2: ATTACK COMPARISON")
    print("="*50)
    
    # Original attack (using original API)
    print("\n--- ORIGINAL ATTACK (Undefended) ---")
    original_attacker = ModelExtractionAttacker()  # From demonstrate_realistic_attack.py
    original_attacker.collect_training_data(num_samples=1000)
    original_attacker.train_surrogate_model()
    
    # Defended attack (using defended API)
    print("\n--- DEFENDED ATTACK (With Defense) ---")
    defended_attacker = DefendedModelExtractionAttacker(defended_api)
    defended_attacker.collect_training_data(num_samples=1000, user_id="malicious_attacker")
    defended_attacker.train_surrogate_model()
    
    # Evaluate surrogate model performance
    original_performance = evaluate_surrogate_performance(original_attacker.surrogate_model, original_api)
    defended_performance = evaluate_surrogate_performance(defended_attacker.surrogate_model, original_api) if defended_attacker.surrogate_model else 0.0
    
    # Display comprehensive results
    print("\n" + "="*70)
    print("DEFENSE EFFECTIVENESS SUMMARY")
    print("="*70)
    
    print(f"LEGITIMATE USER IMPACT:")
    print(f"Success Rate: {legitimate_success_rate:.1%} (Target: >90%)")
    print(f"Accuracy Preserved: {legitimate_accuracy:.1%} (Target: >90%)")
    
    print(f"\nATTACK MITIGATION:")
    print(f"Original Attack Success: {original_performance:.1%}")
    print(f"Defended Attack Success: {defended_performance:.1%}")
    
    # Calculate improvement
    performance_degradation = original_performance - defended_performance
    improvement_factor = performance_degradation / original_performance * 100 if original_performance > 0 else 0
    
    print(f"\nDEFENSE IMPROVEMENT:")
    print(f"Performance Degradation: {performance_degradation:.1%}")
    print(f"Relative Improvement: {improvement_factor:.1f}%")
    
    # Query success comparison
    try:
        original_query_success = len(original_attacker.all_stolen_inputs) / 1000 if hasattr(original_attacker, 'all_stolen_inputs') else 1.0
    except AttributeError:
        original_query_success = 1.0  # Original attacker collected all data successfully
    
    defended_query_success = defended_attacker.successful_queries / 1000 if hasattr(defended_attacker, 'successful_queries') else 0
    
    print(f"\nQUERY SUCCESS RATES:")
    print(f"Original Attack: {original_query_success:.1%}")
    print(f"Defended Attack: {defended_query_success:.1%}")
    
    # Display defense statistics
    print(defended_api.get_defense_statistics())
    
    # Display timing analysis
    print(defended_api.get_timing_analysis())
    
    # Defense objectives assessment
    print("\nDEFENSE OBJECTIVES ASSESSMENT:")
    model_protected = defended_performance < 0.80
    customer_experience = legitimate_success_rate > 0.90 and legitimate_accuracy > 0.90
    attack_detected = defended_api.defense_stats['suspicious_queries'] > 0
    
    print(f"Model Functionality Protected: {'YES' if model_protected else 'NO'}")
    print(f"Customer Experience Preserved: {'YES' if customer_experience else 'NO'}")
    print(f"Attack Detection Working: {'YES' if attack_detected else 'NO'}")
    
    overall_success = model_protected and customer_experience and attack_detected
    print(f"\nOVERALL DEFENSE SUCCESS: {'EFFECTIVE' if overall_success else 'NEEDS IMPROVEMENT'}")

def test_legitimate_users(defended_api, num_queries=100):
    """Test legitimate banking operations"""
    print(f"Simulating {num_queries} legitimate banking queries...")
    
    # Load test data (simulating real check digits)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    # Select random samples (simulating real check processing)
    indices = np.random.choice(len(test_dataset), num_queries, replace=False)
    
    successful_queries = 0
    accuracy_preserved = 0
    
    # Add small delays to simulate realistic banking operations
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        image_batch = image.unsqueeze(0)  # Single image batches (normal for check processing)
        
        # Add realistic delay between queries (0.1-0.5 seconds)
        if i > 0:
            time.sleep(np.random.uniform(0.1, 0.5))
        
        # Process with defended API
        response = defended_api.process_check_digit(
            image_batch, user_id=f"legitimate_bank_{i%5}")  # 5 different bank clients
        
        if response['status'] == 'success':
            successful_queries += 1
            predicted_digit = response['predicted_digit'][0]
            
            # Check if prediction accuracy is preserved
            if predicted_digit == true_label:
                accuracy_preserved += 1
    
    legitimate_success_rate = successful_queries / num_queries
    legitimate_accuracy = accuracy_preserved / successful_queries if successful_queries > 0 else 0
    
    print(f"Success Rate: {legitimate_success_rate:.2%}")
    print(f"Accuracy Preserved: {legitimate_accuracy:.2%}")
    
    return legitimate_success_rate, legitimate_accuracy

def evaluate_surrogate_performance(surrogate_model, original_api, num_samples=500):
    """Evaluate surrogate model agreement with original model"""
    if surrogate_model is None:
        return 0.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    test_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    agreement_count = 0
    total_samples = 0
    
    with torch.no_grad():
        for idx in test_indices:
            image, _ = test_dataset[idx]
            image_batch = image.unsqueeze(0)
            
            # Original model prediction
            original_output = original_api.process_check_digit(image_batch)
            original_pred = original_output['predicted_digit'][0]
            
            # Surrogate model prediction
            surrogate_output = F.softmax(surrogate_model(image_batch), dim=1)
            surrogate_pred = surrogate_output.argmax(dim=1).item()
            
            if original_pred == surrogate_pred:
                agreement_count += 1
            total_samples += 1
    
    agreement_rate = agreement_count / total_samples
    print(f"Surrogate Model Agreement Rate: {agreement_rate:.2%}")
    
    return agreement_rate

if __name__ == "__main__":
    evaluate_defense_effectiveness()