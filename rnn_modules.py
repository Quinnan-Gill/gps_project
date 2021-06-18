import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        #####################################################################
        # Learnable weights for the `input gate`

        self.W_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
        # Learnable weights for `foreget gate`
        self.W_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
        # Learnable weights for `output gate`
        self.W_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
        # Learnable weights for `new memory cell`
        self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
        # Learnable weights for `new memory cell`
        self.W_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))

        if bias:
            self.b_i = nn.Parameter(torch.Tensor(hidden_size))
            self.b_f = nn.Parameter(torch.Tensor(hidden_size))
            self.b_o = nn.Parameter(torch.Tensor(hidden_size))
            self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('b_i', None)
            self.register_parameter('b_f', None)
            self.register_parameter('b_o', None)
            self.register_parameter('b_c', None)
        
        self.count_parameters()
        self.reset_parameters()
        #####################################################################

        return

    def forward(self, x, prev_state):
        ######################################################################        
        if prev_state is None:
            batch = x.shape[0]
            prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
            prev_c = torch.zeros((batch, self.hidden_size), device=x.device)
        else:
            prev_h, prev_c = prev_state
        
        concat_hx = torch.cat((prev_h, x), dim=1)
        i = torch.sigmoid(F.linear(concat_hx, self.W_i, self.b_i))
        f = torch.sigmoid(F.linear(concat_hx, self.W_f, self.b_f))
        o = torch.sigmoid(F.linear(concat_hx, self.W_o, self.b_o))
        c_tilde = torch.tanh(F.linear(concat_hx, self.W_c, self.b_c))
        next_c = f * prev_c + i * c_tilde
        next_h = o * torch.tanh(next_c)
        #####################################################################

        return next_h, next_c

    def reset_parameters(self):
        sqrt_k = (1. / self.hidden_size)**0.5
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-sqrt_k, sqrt_k)
        return

    def extra_repr(self):
        return 'input_size={}, hidden_size={}, bias={}'.format(
            self.input_size, self.hidden_size, self.bias is not True
        )

    def count_parameters(self):
        print(
            'Total Parameters: %d'.format(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
        )
        return
