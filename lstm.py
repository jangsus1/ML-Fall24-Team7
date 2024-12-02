import pandas as pd
import ast
import numpy as np
from glob import glob
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import numpy as np

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

import joblib


classification_classes = {
  "web_browsing": 0, # web
  "ppt": 1, # design tools,
  "figma": 1, # design tools
  "reading": 2, # reading papers
  "finding_mines": 3, # game
  "chess": 3, # game
  "youtube": 4, # youtube
  "chatting": 5, # chatting
}

category_names = []
for label in sorted(classification_classes.values()):
  name = ""
  for tag, l in classification_classes.items():
    if l == label:
      if name != "":
        name += "/"
      name += tag
  if not name in category_names:
    category_names.append(name)
category_names


def process_class_data(file, window_size, window_interval=5, split=0.9):
    df = pd.read_csv(file)
    df = df.ffill().bfill()
    df = df.dropna()
    
    index = int(split * df.shape[0])
    
    train_df = df.iloc[:index]
    test_df = df.iloc[index:]
    
    train_windows = []
    for i in range(0, train_df.shape[0] - window_size + 1, window_interval):
        window = train_df.iloc[i:i+window_size]
        features = window.values.flatten()
        train_windows.append(features)
    
    test_windows = []
    for i in range(0, test_df.shape[0] - window_size + 1, window_interval):
        window = test_df.iloc[i:i+window_size]
        features = window.values.flatten()
        test_windows.append(features)
        
    return train_windows, test_windows

def prepare_data(window_size=50, window_interval=5, resample=True, split=0.8):
    # Process the data for each class
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for tag, label in classification_classes.items():
        files = glob(f'train_data/{tag}_*.csv')
        for file in files:
            # print(f'Processing {file} for class {tag}')
            train_windows, test_windows = process_class_data(file, window_size, window_interval, split=split)
            X_train.extend(train_windows)
            X_test.extend(test_windows)
            y_train.extend([label] * len(train_windows))
            y_test.extend([label] * len(test_windows))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = np.nan_to_num(X_train, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    X_test = np.nan_to_num(X_test, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    # Print per-class counts
    # for label, count in zip(*np.unique(y_train, return_counts=True)):
    #     print(f'Class {label}: {count} samples')

    if resample:
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # print(f'X_train shape: {X_train.shape} -> {X_resampled.shape}')
    # print(f'X_test shape: {X_test.shape}')
    return X_train, X_test, y_train, y_test, scaler


# Prepare the data
window_interval = 2

for window_size in [50, 100, 200, 400]:
    X_train, X_test, y_train, y_test, scaler = prepare_data(window_size, window_interval)

    X_train = X_train.reshape(X_train.shape[0], window_size, -1)
    X_test = X_test.reshape(X_test.shape[0], window_size, -1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the LSTM model
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)  # hn is the last hidden state
            hn = hn[-1]  # Take the last layer's output
            out = self.fc(hn)
            return out

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[2]
    hidden_size = 64
    num_classes = len(np.unique(y_train))
    model = LSTMClassifier(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

    # Evaluate the model
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.numpy())

    # Classification report
    print(classification_report(y_true, y_pred))