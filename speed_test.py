import torch
import torch.nn as nn
import time
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification

# Define LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_classes=10):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn is the last hidden state
        out = hn[-1]  # Take the last layer's output
        out = self.fc1(torch.relu(out))
        out = self.fc2(torch.relu(out))
        return out

# Model initialization
input_size = 28
hidden_size = 128
num_classes = 10
model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# Test data for LSTM
batch_size = 1
sequence_length = 300
# Generate synthetic data for Gradient Boosting and Random Forest
n_classes = 5
n_clusters_per_class = 1  # Reduce clusters per class
n_informative = 4  # Increase informative features

X, y = make_classification(
    n_samples=50,
    n_features=input_size * sequence_length,  # Adjust features based on sequence length
    n_classes=n_classes,
    n_clusters_per_class=n_clusters_per_class,
    n_informative=n_informative,
    random_state=42
)
test_sample = X[0].reshape(1, -1)  # Single test sample

# Histogram Gradient Boosting
hgb = HistGradientBoostingClassifier()
hgb.fit(X, y)
times = []
for i in range(100):
  start_time = time.time()
  hgb.predict(test_sample)
  end_time = time.time()
  times.append(end_time - start_time)
  
inference_time_hgb = sum(times) / len(times)
print(f"Inference time for Histogram Gradient Boosting: {inference_time_hgb:.6f} seconds")

# Random Forest
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X, y)
times = []
for i in range(100):
  start_time = time.time()
  rf.predict(test_sample)
  end_time = time.time()
  times.append(end_time - start_time)
inference_time_rf = sum(times) / len(times)
print(f"Inference time for Random Forest: {inference_time_rf:.6f} seconds")


# Measure inference speed for LSTM
model.eval()
times = []
with torch.no_grad():
  for i in range(100):
    start_time = time.time()
    data = torch.tensor(test_sample, dtype=torch.float32).reshape(batch_size, sequence_length, input_size)
    output = model(data)
    end_time = time.time()
    times.append(end_time - start_time)

inference_time_lstm = sum(times) / len(times)
print(f"Inference time for LSTM (CPU): {inference_time_lstm:.6f} seconds")

# do the GPU test with mps device
device = torch.device("mps")
model.to(device)
data = data.to(device)
model.eval()
times = []
with torch.no_grad():
  for i in range(100):
    start_time = time.time()
    data = torch.tensor(test_sample, dtype=torch.float32, device=device).reshape(batch_size, sequence_length, input_size)
    output = model(data)
    end_time = time.time()
    times.append(end_time - start_time)
  
inference_time_lstm_gpu = sum(times) / len(times)
print(f"Inference time for LSTM (MPS): {inference_time_lstm_gpu:.6f} seconds")

