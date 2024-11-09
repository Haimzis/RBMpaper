import pickle
import numpy as np
import torch
from rbm import RBM


def load_population_pvals(file_path):
    """Load the population pvals from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
    
def forward_pass(v0, rbm_layers):
    """
    Forward pass through a stack of RBM layers to compute the final hidden representation.
    
    Parameters:
    - v0: Initial input vector.
    - rbm_layers: List of RBM layer objects.
    
    Returns:
    - Final hidden representation after passing through all RBM layers.
    """
    v = v0
    for rbm in rbm_layers:
        _, h = rbm.v_to_h(v)
        v = h  # Output of one layer becomes the input to the next
    return v  # Final hidden layer representation


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def soft_thresholding(x, threshold):
    """
    Apply soft thresholding to enforce sparsity.
    
    Parameters:
    - x: Input tensor.
    - threshold: Threshold value for sparsity.
    
    Returns:
    - Thresholded tensor.
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0))


def predict(data, rbm_layers, mode='deterministic', nit=100):
    """
    Predict probabilities or binary labels using the trained RBM layers.

    Parameters:
    - data: Input tensor of shape [n_samples, n_visible].
    - rbm_layers: List of trained RBM layers.
    - mode: 'deterministic' for a single forward pass, 'stochastic' for multiple passes with sampling.
    - nit: Number of iterations for stochastic mode (ignored if mode is 'deterministic').

    Returns:
    - predictions: Tensor of predicted probabilities for each sample.
    """
    n_samples = data.size(0)
    
    if mode == 'deterministic':
        # Perform a single deterministic forward pass through all RBM layers
        hidden_rep = forward_pass(data, rbm_layers)
        predictions = torch.sigmoid(hidden_rep)  # Apply sigmoid to map to probabilities

    elif mode == 'stochastic':
        # Perform multiple stochastic passes to get averaged probabilities
        probs = torch.zeros(n_samples, rbm_layers[-1].n_hidden)
        for _ in range(nit):
            # Use a fresh copy of data for each stochastic forward pass
            hidden_rep = data
            for rbm in rbm_layers:
                poshidprobs, _ = rbm.v_to_h(hidden_rep)
                hidden_rep = torch.bernoulli(poshidprobs)  # Sample binary hidden states
            probs += torch.sigmoid(hidden_rep)  # Accumulate probabilities
        
        # Average probabilities across iterations
        predictions = probs / nit

    return predictions


# Parameters
n_visible = 16  # Example dimension of input
n_hidden_list = [32, 32, 16]  # Hidden layer sizes for each RBM layer
learning_rate = 0.001
epochs = 10  # Number of training epochs per layer

# Initialize RBM layers
rbm_layers = [RBM(n_visible if i == 0 else n_hidden_list[i-1], n_hidden_list[i], learning_rate=learning_rate) for i in range(len(n_hidden_list))]

# Sample data
data = load_population_pvals('real_population_pvals_10kb.pkl')
data = torch.tensor(np.array(data), dtype=torch.float32)

# Train each RBM layer in a greedy fashion
for layer_idx, rbm in enumerate(rbm_layers):
    print(f"Training RBM layer {layer_idx + 1}/{len(rbm_layers)}")
    optimizer = torch.optim.SGD(rbm.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data:
            batch = batch.view(1, -1)
            loss = rbm.contrastive_divergence(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(data)}")
    
    # After training, compute hidden representation for the next layer's input
    with torch.no_grad():
        data = forward_pass(data, [rbm])

# # Final representations for the trained RBM layers can now be used for further tasks
# # Make predictions (probabilities) in deterministic mode
# predictions_det = predict(test_data, rbm_layers, mode='deterministic')
# print("Deterministic predictions:", predictions_det)

# # Make predictions (probabilities) in stochastic mode
# predictions_stoch = predict(test_data, rbm_layers, mode='stochastic', nit=100)
# print("Stochastic predictions:", predictions_stoch)