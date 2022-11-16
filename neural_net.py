from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, fc_layer_size, num_layers_hidden, dropout):
      super(NeuralNetwork, self).__init__()

      if num_layers_hidden >= 1:
        #first layer
        hidden_linear_relu_stack = [nn.Linear(input_dim, fc_layer_size),  nn.Dropout(p=dropout), nn.ReLU()]

        for _ in range(num_layers_hidden-1):
          #inner hidden layers
          hidden_linear_relu_stack.append(nn.Linear(fc_layer_size, fc_layer_size))
          hidden_linear_relu_stack.append(nn.Dropout(p=dropout))
          hidden_linear_relu_stack.append(nn.ReLU())

        #ouput layer 
        hidden_linear_relu_stack.append(nn.Linear(fc_layer_size, 1))

      else:
        hidden_linear_relu_stack = [nn.Linear(input_dim, 1)]

      self.hidden_linear_relu_stack = nn.Sequential(*hidden_linear_relu_stack)

    def forward(self, x):
      return self.hidden_linear_relu_stack(x).flatten()