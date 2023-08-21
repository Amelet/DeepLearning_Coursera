# Neural network model

The general methodology to build a Neural Network is to:

### Define the neural network structure ( # of input units,  # of hidden units, etc). 
Define three variables:

- *n_x:* the size of the input layer
- *n_h:* the size of the hidden layer (set this to 4) 
- *n_y:* the size of the output layer

### Initialize the model's parameters
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrices with random values. 
- Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vectors as zeros. 
- Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.

```python
W1 -- weight matrix of shape (n_h, n_x)
b1 -- bias vector of shape (n_h, 1)
W2 -- weight matrix of shape (n_y, n_h)
b2 -- bias vector of shape (n_y, 1)

W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))
```
### Loop:
- Implement forward propagation
Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).

### Compute loss
Loss function:
$$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

```python
   # Compute the cross-entropy cost
    logprobs1 = np.multiply(np.log(A2), Y)
    logprobs2 = np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs1 + logprobs2)/m
```


### Implement backward propagation to get the gradients
```python
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
 
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    tanh_derivative_Z1 = (1 - np.power(A1, 2))
    dZ1 = np.dot(W2.T, dZ2) * tanh_derivative_Z1
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
```

### Update parameters (gradient descent)
```python
    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
```


**Notes**:
- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
- Regularization lets you use very large models (such as n_h = 50) without much overfitting. 


