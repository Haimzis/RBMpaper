import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1, learning_rate=0.1):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # Number of Gibbs sampling steps
        self.lr = learning_rate

        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)  # Weight matrix
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # Hidden layer bias
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # Visible layer bias

    def sample_from_p(self, p):
        """Sample binary values from a Bernoulli distribution."""
        return torch.bernoulli(p)

    def v_to_h(self, v):
        """Compute the probability of hidden units being 1 given visible units."""
        p_h_given_v = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h_given_v, self.sample_from_p(p_h_given_v)

    def h_to_v(self, h):
        """Compute the probability of visible units being 1 given hidden units."""
        p_v_given_h = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v_given_h, self.sample_from_p(p_v_given_h)

    def contrastive_divergence(self, v0):
        """Perform one step of contrastive divergence."""
        # Positive phase
        p_h0, h0 = self.v_to_h(v0)
        
        # Negative phase (Gibbs sampling)
        vk = v0
        for _ in range(self.k):
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)

        # Compute gradients
        positive_grad = torch.matmul(p_h0.t(), v0)
        negative_grad = torch.matmul(hk.t(), vk)

        # Update parameters
        self.W.data += self.lr * (positive_grad - negative_grad) / v0.size(0)
        self.v_bias.data += self.lr * torch.sum(v0 - vk, dim=0) / v0.size(0)
        self.h_bias.data += self.lr * torch.sum(p_h0 - hk, dim=0) / v0.size(0)

        # Compute reconstruction error
        loss = torch.mean((v0 - vk) ** 2)
        return loss
