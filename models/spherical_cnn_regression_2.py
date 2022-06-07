import torch
import torch.nn as nn

import gelib_torchC as gelib
from gelib_torchC import *

class Spherical_CNN_Regression(torch.nn.Module):
    def __init__(self, num_layers, input_tau, hidden_tau, maxl, diag_cg = True, has_normalization = True, device = 'cuda'):
        super(Spherical_CNN_Regression, self).__init__()
        
        # Hyper-parameters
        self.num_layers = num_layers
        self.input_tau = input_tau
        self.hidden_tau = hidden_tau
        self.maxl = maxl
        self.diag_cg = diag_cg
        self.has_normalization = has_normalization
        self.num_learnable_params = 0
        self.device = device

        # Check the tau
        # self.maxl = max(len(input_tau), len(hidden_tau)) - 1
        assert(len(hidden_tau) <= self.maxl + 1)
        for l in range(len(input_tau)):
            assert(input_tau[l] > 0)
        for l in range(len(hidden_tau)):
            assert(hidden_tau[l] > 0)

        # Keep track of tau
        tau = []

        # Create learnable weights for input to hidden
        self.W_input = nn.ParameterList()
        for l in range(len(input_tau)):
            if l < len(hidden_tau):
                weight = nn.Parameter(torch.randn(input_tau[l], hidden_tau[l], 2).to(device = self.device))
                self.W_input.append(weight)
                tau.append(hidden_tau[l])

                # Parameter register
                self.register_parameter(
                        name = 'W_input_l_' + str(l),
                        param = weight
                )
                self.num_learnable_params += 1
            else:
                break

        # Create learnable weights for hidden to hidden
        self.W_hidden = []
        for layer in range(num_layers):
            if self.diag_cg == False:
                next_tau = CGproductType(tau, tau, self.maxl)
            else:
                next_tau = DiagCGproductType(tau, tau, self.maxl)

            tau = []
            W = nn.ParameterList()
            for l in range(len(next_tau)):
                if l < len(hidden_tau) and next_tau[l] > 0:
                    weight = nn.Parameter(torch.randn(next_tau[l], hidden_tau[l], 2).to(device = self.device))
                    W.append(weight)
                    tau.append(hidden_tau[l])

                    # Parameter register
                    self.register_parameter(
                            name = 'W_hidden_layer_' + str(layer) + '_l_' + str(l),
                            param = weight
                    )
                    self.num_learnable_params += 1
                else:
                    break
            self.W_hidden.append(W)

        # Top layer
        total_size = 0
        for l in range(len(tau)):
            total_size += tau[l]

        print('Total size:', total_size)
        self.top_layer_1 = nn.Linear(total_size, 256).to(device = self.device)
        self.top_layer_2 = nn.Linear(256, 1).to(device = self.device)

    def tensor_normalization(self, inputs):
        outputs = []
        for i in range(len(inputs)):
            tensor = torch.view_as_complex(inputs[i])
            norm = torch.norm(tensor, dim = 1, p = 2)
            norm = 1.0 / norm
            out = torch.view_as_real(torch.einsum('bij,bj->bij', tensor, norm))
            outputs.append(out)
        return outputs

    def forward(self, inputs):

        # +---------------------------------------------------------------------+
        # | Convert the inputs (SO3vec) to the right device (e.g., cpu or cuda) |
        # +---------------------------------------------------------------------+

        batch = inputs.parts[0].size(0)
        inputs_ = gelib.SO3vec.zeros(batch, self.input_tau)
        for l in range(len(self.input_tau)):
            inputs_.parts[l] = inputs.parts[l].detach().to(device = self.device)
            # inputs_.parts[l].requires_grad_()
        inputs = inputs_

        # +-----------------+
        # | Input to hidden |
        # +-----------------+
        
        # Mixing channels
        hidden = []
        for l in range(len(self.W_input)):
            first = torch.view_as_complex(inputs.parts[l]).to(device = self.device)
            second = torch.view_as_complex(self.W_input[l])
            hidden.append(torch.view_as_real(torch.matmul(first, second)).to(device = self.device))
        
        # Tensor normalization
        if self.has_normalization == True:
            hidden = self.tensor_normalization(hidden)

        # Create the SO3vec
        hidden = gelib.SO3vec(hidden)

        # +------------------+
        # | Hidden to hidden |
        # +------------------+

        # For each layer
        for layer in range(self.num_layers):
            # CG product
            if self.diag_cg == False:
                product = gelib.CGproduct(hidden, hidden, self.maxl)
            else:
                product = gelib.DiagCGproduct(hidden, hidden, self.maxl)

            # Mixing channels
            next_hidden = []
            for l in range(len(self.W_hidden[layer])):
                first = torch.view_as_complex(product.parts[l])
                second = torch.view_as_complex(self.W_hidden[layer][l])
                next_hidden.append(torch.view_as_real(torch.matmul(first, second)))
            
            # Tensor normalization
            if self.has_normalization == True and layer + 1 < self.num_layers:
                next_hidden = self.tensor_normalization(next_hidden)

            # Create the SO3vec
            hidden = gelib.SO3vec(next_hidden)

        # +------------------+
        # | Hidden to output |
        # +------------------+

        outputs = []
        for l in range(len(hidden.parts)):
            tensor = torch.view_as_complex(hidden.parts[l])
            norm = torch.norm(tensor, dim = 1, p = 2)
            outputs.append(norm)

        # Concatenate 
        outputs = torch.cat(outputs, dim = 1)

        # Multilayer Perceptron for the top layer
        outputs = torch.tanh(self.top_layer_1(outputs))

        return self.top_layer_2(outputs)

