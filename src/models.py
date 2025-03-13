import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x
    
# ---------------------------------------------------------------------------
# LearnedUpdate:
#
# Computes:
#   v_t = (α₀ + α₁·t + ... + α_q·t^q) · tanh(MLP([x_t, f(x_t), ∇f(x_t)])) · ρ^t,
#
# where:
#   - α are learnable parameters (transformed via softplus to ensure positivity),
#   - The MLP processes the concatenated vector [x_t, f(x_t), ∇f(x_t)],
#   - ρ is a fixed constant (e.g. 0.99) that scales the update over time.
# ---------------------------------------------------------------------------
class LearnedUpdate(nn.Module):
    def __init__(self, d, q, rho, hidden_size1, hidden_size2):
        """
        d: dimension of parameter vector x.
        q: order of the polynomial (i.e. there are q+1 coefficients). For example, q=2 uses [1, t, t²]
        rho: fixed discount factor (must be less than 1).
        """
        super(LearnedUpdate, self).__init__()
        # The MLP takes a vector of size 2*d + 1 ([x, f(x), grad f(x)]) and outputs a vector of size d.
        self.mlp = SimpleMLP(input_size=2 * d + 1, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=d)
        # Initialize raw alpha parameters; they will be transformed to be positive.
        self.alpha_raw = nn.Parameter(torch.full((q + 1,), -2.0))

        self.q = q
        self.rho = rho  # fixed (non-learnable) discount factor
    
    def get_dimension(self):
        return (self.mlp.layer1.in_features - 1 )/ 2
    
    def forward(self, x, loss_val, grad, t):
        """
        x: tensor of shape (1, d)
        loss_val: tensor of shape (1, 1)
        grad: tensor of shape (1, d)
        t: current time-step (an integer or float)
        """
        # Concatenate x, loss, and grad to form the input for the MLP.
        input_vec = torch.hstack([x, loss_val, grad])   # shape: (1, 2*d + 1)
        mlp_out = torch.tanh(self.mlp(input_vec))       # shape: (1, d)

        # Compute the polynomial factor: [1, t, t^2, ..., t^q] dot α (with α transformed to be positive)
        alpha = F.softplus(self.alpha_raw)  # shape: (q+1,)
        poly_terms = torch.tensor([t ** i for i in range(self.q + 1)], device=x.device, dtype=x.dtype)  # shape: (q+1,)
        poly_factor = torch.dot(alpha, poly_terms)  # scalar

        # Multiply the MLP output by the polynomial factor and a fixed decay factor ρ^t.
        v_t = poly_factor * mlp_out * (self.rho ** t)  # shape: (1, d)
        return v_t