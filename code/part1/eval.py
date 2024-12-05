"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()

    # Obtenir les ensembles de test et leurs cibles pour une cardinalité donnée
    X_card_test = X_test[i]  # Multisets avec une cardinalité spécifique
    y_card_test = y_test[i]  # Sommes correspondantes
    
    # Convertir en tenseurs PyTorch
    X_card_tensor = torch.tensor(X_card_test, dtype=torch.long, device=device)
    y_card_tensor = torch.tensor(y_card_test, dtype=torch.float32, device=device)
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        x_batch = X_card_tensor[j:j + batch_size]
        y_batch = y_card_tensor[j:j + batch_size]
        
        # Prédictions avec DeepSets
        with torch.no_grad():
            y_pred_batch_deepsets = deepsets(x_batch)
        y_pred_deepsets.append(y_pred_batch_deepsets)
        
        # Prédictions avec LSTM
        with torch.no_grad():
            y_pred_batch_lstm = lstm(x_batch)
        y_pred_lstm.append(y_pred_batch_lstm)
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    y_pred_deepsets_np = y_pred_deepsets.cpu().numpy()
    y_pred_lstm_np = y_pred_lstm.cpu().numpy()
    y_card_test_np = y_card_tensor.cpu().numpy()
    
    acc_deepsets = accuracy_score(np.round(y_card_test_np), np.round(y_pred_deepsets_np))
    mae_deepsets = mean_absolute_error(y_card_test_np, y_pred_deepsets_np)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(np.round(y_card_test_np), np.round(y_pred_lstm_np))
    mae_lstm = mean_absolute_error(y_card_test_np, y_pred_lstm_np)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
# Visualisation des précisions
plt.figure(figsize=(10, 6))
plt.plot(cards, results['deepsets']['acc'], label='DeepSets Accuracy')
plt.plot(cards, results['lstm']['acc'], label='LSTM Accuracy')
plt.xlabel('Cardinality of the Input Sets')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Cardinality')
plt.legend()
plt.savefig('precisions.png')
plt.show()

# Visualisation des erreurs absolues moyennes
plt.figure(figsize=(10, 6))
plt.plot(cards, results['deepsets']['mae'], label='DeepSets MAE')
plt.plot(cards, results['lstm']['mae'], label='LSTM MAE')
plt.xlabel('Cardinality of the Input Sets')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE vs Cardinality')
plt.legend()
plt.savefig('erreurs.png')
plt.show()
##################
