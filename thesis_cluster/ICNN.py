import torch
import torch.nn as nn
import torch.nn.functional as F

class Kantorovich_Potential(nn.Module):
    ''' 
        Modelling the Kantorovich potential as Input convex neural network (ICNN)
        input: y
        output: z = h_L
        Architecture: h_1     = ReLU^2(A_0 y + b_0)
                      h_{l+1} =   ReLU(A_l y + b_l + W_{l-1} h_l)
        Constraint: W_l > 0
    '''
    def __init__(self, input_size, hidden_size_list):
        super(Kantorovich_Potential, self).__init__()
        self.input_size = input_size
        self.num_hidden_layers = len(hidden_size_list)
        
        # list of matrices that interacts with input
        self.A = nn.ParameterList([nn.Parameter(torch.rand(self.input_size, hidden_size_list[k]) * 0.1) for k in range(self.num_hidden_layers)])
        
        # list of bias vectors at each hidden layer 
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1, hidden_size_list[k])) for k in range(self.num_hidden_layers)])
        
        # list of matrices between consecutive layers
        self.W = nn.ParameterList([nn.Parameter(torch.rand(hidden_size_list[k-1], hidden_size_list[k]) * 0.1) for k in range(1, self.num_hidden_layers)])

    def forward(self, input_y):
        # Using ReLU Squared
        z = F.leaky_relu(torch.matmul(input_y, self.A[0]) + self.b[0], negative_slope=0.2)
        z = z.pow(2)

        for k in range(1, self.num_hidden_layers):
            z = F.leaky_relu(torch.matmul(input_y, self.A[k]) + self.b[k] + torch.matmul(z, self.W[k-1]), negative_slope=0.2)

        return z
    
    def positive_constraint_loss(self):
        """ regularization term for g """
        return torch.sum(torch.tensor([torch.sum(torch.relu(-w)**2) for w in self.W]))
    
    # During training enforce positive weights
    def enforce_positive_weights(self):
        for _ in range(len(self.W)):
            self.W[_] = torch.relu(self.W[_])
