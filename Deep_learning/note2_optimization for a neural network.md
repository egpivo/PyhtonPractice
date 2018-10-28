
# Note 2: optimization for a neural network
WT Wang                     
May, 03, 2017

------

## Loss function
After fitting the model based on the certain weights, we can also calculate the prediction errors to see how accurate the predictions are to the actual value. The prediction error is , $e_i({\bf{w}}) = y_i - \hat{y}_i({\bf{w}})$. However, how can we know the current predictions are the best ones? We can measure these by a loss function . For example, a squared error loss function is $$L_1({\bf{w}}) = \sum_{i=1}^n e^2_i({\bf{w}});$$ an absolute error loss functionis $$L_2({\bf{w}}) = \sum_{i=1}^n \left|e_i({\bf{w}})\right|.$$ That is, a lower loss function value means a better model performance.

### Practice: toy example

Set two candidate sets of the weights as 
    - 1st set: 
    
${\bf{\omega}}_{11} = \left[ \begin{array}{cc}
-5 & 5 \\
-1 & 1\\
\end{array} \right]$, ${\bf{\gamma}}_1 = \left[ \begin{array}{c}
3  \\
7\\
\end{array} \right]$
     - 2nd set: 
    
${\bf{\omega}}_{12} = \left[ \begin{array}{cc}
1 & 1 \\
2 & -1.5\\
\end{array} \right]$, ${\bf{\gamma}}_2 = \left[ \begin{array}{c}
1.9  \\
3.5\\
\end{array} \right]$.


The input dataset is ${\bf{X}}= \left[ \begin{array}{cc}
0 & 1\\
5 & 7\\
6 & -2\\
10 & 11\\
\end{array} \right]$, and the response is  ${\bf{y}}= \left[ \begin{array}{c}
7\\
5\\
2\\
20\\
\end{array} \right]$. Use `predict_with_one_layer()` to calculate the predicitons, and the squared error loss function to measure the performance. The result below shows that the second candidate set of weights perfoms better than the first one.


```python
from sklearn.metrics import mean_squared_error
import numpy as np
# import the functions in note1
from note1 import *

# Create model_output_1 
model_output_1 = []
# Create model_output_2
model_output_2 = []

weights_1 = {'node_0': np.array([-5, 5]),
            'node_1': np.array([-1, 1]),
            'output': np.array([7, 5])}

weights_2 = {'node_0': np.array([1, 1]),
            'node_1': np.array([2, -1.5]),
            'output': np.array([1.9, 3.5])}

input_data = [np.array([0, 1]),
              np.array([5, 7]),
              np.array([6, -2]),
              np.array([10, 11])]

target_actuals = [7, 5, 2, 20]

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_1.append(predict_with_one_layer(row, weights_1))
    
    # Append prediction to model_output_1
    model_output_2.append(predict_with_one_layer(row, weights_2))

# Calculate the mean squared error for model_output_0: mse_0
mse_1 = mean_squared_error(model_output_1, target_actuals)

# Calculate the mean squared error for model_output_1: mse_1
mse_2 = mean_squared_error(model_output_2, target_actuals)

# Print mse_0 and mse_1
print("Mean squared error with weights_1: %f" %mse_1)
print("Mean squared error with weights_2: %f" %mse_2)
```

    Mean squared error with weights_1: 1779.500000
    Mean squared error with weights_2: 1188.020625


- Note: we can see that the sencond weight performs better.

## Optimization
Here, we want to find the weights that give the lowest value for the loss function.

### Gradient decent (GD)
Simply, we can apply the gradient descent to address this problem. Gradient descent is a first-order iterative optimization algorithm, that is, the solution is computed along with the paths of the slope of loss function with respect to the weights. More detail can be found in [here](https://en.wikipedia.org/wiki/Gradient_descent). 


```python
import numpy as np
from sklearn.metrics import mean_squared_error

def gradientDescent(input_data, target, weights, learning_rate = 0.01, n_updates = 5):
    
    # Iterate over the number of updates
    for i in range(n_updates):
        # Calculate the predictions: preds
        preds = (weights * input_data).sum()
        # Calculate the error: error
        error = target - preds
        # Calculate the slope: slope
        slope = 2 * input_data * error
        # Update the weights: weights
        weights = weights + slope * learning_rate

        mse = (error) **2
          
        print("Iteration %d -- loss: %f" % (i+1, mse))
     
    return weights
```

### Practice: toy example

Set a initial vector of weights as   
${\bf{\omega}} = \left[ \begin{array}{c}
0 \\
0 \\
0
\end{array} \right]$, the input dataset is ${\bf{X}}= \left[ \begin{array}{c}
3\\
1\\
5
\end{array} \right]$, and the response is  $ y=8$. Use `gradientDescent()` to calculate the estimation of weights.


```python
weights = np.array([0, 0, 0])     
input_data = np.array([3, 1, 5])
target = -8
new_weight = gradientDescent(input_data, target, weights)
print(new_weight)
```

    Iteration 1 -- loss: 64.000000
    Iteration 2 -- loss: 5.760000
    Iteration 3 -- loss: 0.518400
    Iteration 4 -- loss: 0.046656
    Iteration 5 -- loss: 0.004199
    [-0.684048 -0.228016 -1.14008 ]


- Note: we can see that the loss decreases as the number of iteration gets larger.

### Stochastic gradient descent (SGD)
- The process of SGD is:
     1. It is common to calculate slopes on only a subset of the randomly shuffled data (‘batch’)
     2. Use a different batch of data to calculate the next update
     3. Start over from the beginning once all data is used

- The algorithm:
       1. Randomly shuffle the data
       2. Split m subsets
       3. Do{
             for i = 1, ..., m:
                 \omega_i := \omega_i - learning rate * slope
          } until converge

Remark: SGD usually converges faster than GD with a mild convergence rate. More detail can be found [here](http://cs229.stanford.edu/notes/cs229-notes1.pdf).  


## Backpropagation

The process of backpropagation is:
1. Start at some random set of weights
2. Use forward propagation to make a prediction
3. Use backward propagation to calculate the slope of the loss function w.r.t each weight
4. Multiply that slope by the learning rate, and subtract from the current weights
5. Keep going with that cycle until we get to a flat part

Remark: More detail can be found [here](https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf).  

