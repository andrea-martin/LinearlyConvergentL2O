import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class TwoLayerLSTM(nn.Module):
    def __init__(self, input_sz, output_sz,  hidden_sz=20, device=torch.device('cpu')):
        super().__init__()

        self.hidden_sz = hidden_sz

        std = 0.01

        self.hidden1_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.cell1_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.hidden2_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))
        self.cell2_0 = nn.Parameter((torch.randn(1, hidden_sz, device=device) * std ))

        self.recurs = nn.LSTMCell(input_sz, hidden_sz, device=device)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz, device=device)
        self.output = nn.Linear(hidden_sz, output_sz, device=device)

        self.hidden1 = torch.zeros(1, hidden_sz, device=device)
        self.cell1 = torch.zeros(1, hidden_sz, device=device)
        self.hidden2 = torch.zeros(1, hidden_sz, device=device)
        self.cell2 = torch.zeros(1, hidden_sz, device=device)

    def forward(self, inp, hidden1, cell1, hidden2, cell2):

        (hidden1_, cell1_) = self.recurs(inp, (hidden1, cell1))
        (hidden2_, cell2_) = self.recurs2(hidden1_, (hidden2, cell2))

        return self.output(hidden2_), (hidden1_, cell1_), (hidden2_, cell2_)





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
    def __init__(self, d, q, rho, hidden_sizes, architecture='mlp'):
        """
        d: dimension of parameter vector x.
        q: order of the polynomial (i.e. there are q+1 coefficients). For example, q=2 uses [1, t, t²]
        rho: fixed discount factor (must be less than 1).
        """
        super(LearnedUpdate, self).__init__()
        # The MLP takes a vector of size 2*d + 1 ([x, f(x), grad f(x)]) and outputs a vector of size d.
        # self.mlp = SimpleMLP(input_size=2 * d + 1, hidden_sizes=hidden_sizes, output_size=d)
        
        self.lstm = TwoLayerLSTM(input_sz=2*d+1, output_sz=d, hidden_sz=20) 
        
        # Initialize raw alpha parameters; they will be transformed to be positive.
        self.alpha_raw = nn.Parameter(torch.full((q + 1,), -2.0))

        self.q = q
        self.rho = rho  # fixed (non-learnable) discount factor

        self.architecture = architecture
    
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
            # Compute separate alpha values
            alpha_x = torch.norm(x_batch[i], p=float('inf'))  # Max norm of x_batch[i]
            alpha_loss = torch.norm(loss_batch[i], p=float('inf'))  # Max norm of loss_batch[i]
            alpha_grad = torch.norm(grad_batch[i], p=float('inf'))  # Max norm of grad_batch[i]

            # Prevent division by zero (if any alpha is zero, set it to 1 to avoid NaNs)
            alpha_x = alpha_x if alpha_x > 0 else torch.tensor(1.0, device=x_batch[i].device)
            alpha_loss = alpha_loss if alpha_loss > 0 else torch.tensor(1.0, device=loss_batch[i].device)
            alpha_grad = alpha_grad if alpha_grad > 0 else torch.tensor(1.0, device=grad_batch[i].device)

            # Scale inputs
            x_scaled = x_batch[i] / alpha_x
            loss_scaled = loss_batch[i] / alpha_loss
            grad_scaled = grad_batch[i] / alpha_grad

            # Stack scaled inputs and pass through MLP
            input_vec = torch.vstack([x_scaled, loss_scaled, grad_scaled])  # Shape: (2d+1, 1)
            # mlp_output = torch.tanh(self.mlp(input_vec.squeeze(-1)))  # Shape: (d,)

            # lstm_output, (self.lstm.hidden1, self.lstm.cell1), (self.lstm.hidden2, self.lstm.cell2) = self.lstm.forward(input_vec.T, self.lstm.hidden1, self.lstm.cell1, self.lstm.hidden2, self.lstm.cell2)  # Shape: (1, d)
            # lstm_output = torch.tanh(lstm_output).squeeze(0)
            # Update LSTM states without modifying them in-place
            
            
            if t == 0:
                if self.architecture == 'lstm':
                    lstm_output, (self.lstm.hidden1, self.lstm.cell1), (self.lstm.hidden2, self.lstm.cell2) = self.lstm.forward(input_vec.T, self.lstm.hidden1_0, self.lstm.cell1_0, self.lstm.hidden2_0, self.lstm.cell2_0)
                    lstm_output = torch.tanh(lstm_output).squeeze(0)
                else:
                    pass
            else:
                if self.architecture == 'lstm':
                    lstm_output, (self.lstm.hidden1, self.lstm.cell1), (self.lstm.hidden2, self.lstm.cell2) = self.lstm.forward(input_vec.T, self.lstm.hidden1, self.lstm.cell1, self.lstm.hidden2, self.lstm.cell2)
                    lstm_output = torch.tanh(lstm_output).squeeze(0)
                else:
                    pass
                    
            
            # lstm_output, (hidden1_new, cell1_new), (hidden2_new, cell2_new) = self.lstm.forward(
            #     input_vec.T, self.lstm.hidden1, self.lstm.cell1, self.lstm.hidden2, self.lstm.cell2
            # )
            # # Update the LSTM states after computation
            # self.lstm.hidden1, self.lstm.cell1 = hidden1_new, cell1_new
            # self.lstm.hidden2, self.lstm.cell2 = hidden2_new, cell2_new
            # lstm_output = torch.tanh(lstm_output).squeeze(0)

            # Rescale output
            if self.architecture == 'lstm':
                output = lstm_output * alpha_x
            else:
                pass
                # output = mlp_output * alpha_x  # Rescale using alpha_x (assuming x is the dominant feature)

            results.append(output) 

        direction_batch = torch.stack(results, dim=0)  # Shape: (batch_size, 2d+1, d)

        # Compute the polynomial factor
        alpha = F.softplus(self.alpha_raw)
        poly_terms = torch.tensor([t ** j for j in range(self.q + 1)], device=x_batch.device, dtype=x_batch.dtype)
        poly_factor = torch.dot(alpha, poly_terms)

        # Apply update rule
        v_t_batch = poly_factor * direction_batch * (self.rho ** t)  # Shape: (batch_size, 2d+1, d)

        return v_t_batch