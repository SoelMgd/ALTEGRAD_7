"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        ##################
        
        embedded = self.embedding(x) 
        transformed = self.tanh(self.fc1(embedded))  
        summed = transformed.sum(dim=1)
        x = self.fc2(summed)
        ##################
        
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
    
        ##################
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: (1, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        x = self.fc(last_hidden)  # Shape: (batch_size, 1)
        ##################
        
        return x.squeeze()
