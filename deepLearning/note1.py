import numpy as np
def reLU(input):
    # Calculate the value for the output of the reLU function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)


def softplus(input):
    from math import exp, log
    # Calculate the value for the output of the softplus function: output
    output = log(1+ exp(input))
    
    # Return the value just calculated
    return(output)


def predict_with_one_layer(input_data_row, weights):
    # Calculate node 0 value
    node_0_input =  (input_data_row * weights['node_0']).sum()
    node_0_output = reLU(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = reLU(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = reLU(input_to_final_layer)
    
    # Return model output
    return(model_output)


def predict_with_two_layers(input_data, weights):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = reLU(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = reLU(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = reLU(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = reLU(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)