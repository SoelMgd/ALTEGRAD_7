"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
        # Generate training multisets
    X_train = []
    y_train = []

    for _ in range(n_train):
        # Randomly select the cardinality of the multiset (1 to 10)
        cardinality = np.random.randint(1, max_train_card + 1)
        # Generate random digits from {1, ..., 10}
        multiset = np.random.randint(1, 11, size=cardinality)
        # Pad with zeros to ensure all multisets have the same length (10)
        padded_multiset = np.pad(multiset, (max_train_card - len(multiset), 0), mode='constant')
        X_train.append(padded_multiset)
        y_train.append(multiset.sum())  # Compute the target as the sum of the digits
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    n_test_samples = 200000
    X_test = []
    y_test = []

    # Generate test multisets with varying cardinalities (5 to 100)
    for cardinality in range(5, 101, 5):  # Step size is 5
        for _ in range(n_test_samples // ((100 - 5) // 5 + 1)):  # 10,000 samples per cardinality
            # Generate random digits from {1, ..., 10}
            multiset = np.random.randint(1, 11, size=cardinality)
            X_test.append(multiset)
            y_test.append(multiset.sum())  # Compute the target as the sum of the digits
    ##################

    return X_test, y_test
