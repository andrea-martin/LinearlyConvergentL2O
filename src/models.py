import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init



class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleMLP, self).__init__()
        
        # Create the layers dynamically
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)

        # Call custom initializer function
        self._initialize_weights()

    def forward(self, x):
        # Pass through all layers with ReLU activations
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """ Custom weight initialization function """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
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
    def __init__(self, d, q, rho, hidden_sizes):
        """
        d: dimension of parameter vector x.
        q: order of the polynomial (i.e. there are q+1 coefficients). For example, q=2 uses [1, t, t²]
        rho: fixed discount factor (must be less than 1).
        """
        super(LearnedUpdate, self).__init__()
        # The MLP takes a vector of size 2*d + 1 ([x, f(x), grad f(x)]) and outputs a vector of size d.
        self.mlp = SimpleMLP(input_size=2 * d + 1, hidden_sizes=hidden_sizes, output_size=d)
        
        # Initialize raw alpha parameters; they will be transformed to be positive.
        self.alpha_raw = nn.Parameter(torch.full((q + 1,), -13.0))

        self.q = q
        self.rho = rho  # fixed (non-learnable) discount factor
    
    def forward(self, x_batch, loss_batch, grad_batch, t):
        """
        x_batch: tensor of shape (batch_size, d, 1)
        loss_batch: tensor of shape (batch_size, 1, 1)
        grad_batch: tensor of shape (batch_size, d, 1)
        t: current time-step (an integer or float)
        """

        batch_size = x_batch.shape[0]  
        results = []

        for i in range(batch_size):
            input_vec = torch.vstack([x_batch[i], loss_batch[i], grad_batch[i]])  # Shape: (2d+1, 1)
            output = torch.tanh(self.mlp(input_vec.squeeze(-1)))  # Shape: (2d+1, d)
            results.append(output)

        direction_batch = torch.stack(results, dim=0)  # Shape: (batch_size, 2d+1, d)

        # Compute the polynomial factor
        alpha = F.softplus(self.alpha_raw)
        poly_terms = torch.tensor([t ** i for i in range(self.q + 1)], device=x_batch.device, dtype=x_batch.dtype)
        poly_factor = torch.dot(alpha, poly_terms)

        # Apply update rule
        v_t_batch = poly_factor * direction_batch * (self.rho ** t)  # Shape: (batch_size, 2d+1, d)
        return v_t_batch

        # # Concatenate x, loss, and grad to form the input for the MLP.
        # input_vec_batch = torch.hstack([x_batch, loss_batch, grad_batch])   # shape: (batch_size, 2*d + 1,1)
        # direction_batch = torch.tanh(self.mlp(input_vec_batch.squeeze(-1)))       # shape: (batch_size, d)

        # # Compute the polynomial factor: [1, t, t^2, ..., t^q] dot α (with α transformed to be positive)
        # alpha = F.softplus(self.alpha_raw)  # shape: (q+1,)
        # poly_terms = torch.tensor([t ** i for i in range(self.q + 1)], device=x_batch.device, dtype=x_batch.dtype)  # shape: (q+1,)
        # poly_factor = torch.dot(alpha, poly_terms)  # scalar

        # # Multiply the MLP output by the polynomial factor and a fixed decay factor ρ^t.
        # v_t_batch = poly_factor * direction_batch * (self.rho ** t)  # shape: (1, d)
        # return v_t_batch