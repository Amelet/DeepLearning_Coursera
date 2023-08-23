## Initialize the parameters for a two-layer network and for an $L$-layer neural network.
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
```python
    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
```

## Implement the forward propagation module.
- Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).

```python
Z = np.dot(W, A) + b
```

- We give you the ACTIVATION function (relu/sigmoid).
- Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.

```python
        Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
```


- Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.

```python
        A, cache = linear_activation_forward(A_prev, WL, bL, activation='relu')
        caches.append(cache)
```


## Compute the loss.

```python
    log_probs1 = np.dot(Y, np.log(AL).T)
    log_probs2 = np.dot((1-Y), np.log(1-AL).T)
    cost = -1/m*(np.sum(log_probs1+log_probs2))
```

## Implement the backward propagation module (denoted in red in the figure below).

- Complete the LINEAR part of a layer's backward propagation step.
```python
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = 1/m * (np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)
```

- We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) 
- Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.

```python
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
```

- Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function

```python
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = np.divide(Y, AL) - np.divide(1-Y, 1-AL)

    
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA"+str(L-1)], grads["dWL"], grads["dbL"] = linear_activation_backward(dAL, current_cache, "sigmoid")

    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        #print(L, l)
        #print(grads.keys())
        #print("dA" + str(l + 1))
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    ```


## Finally update the parameters.
```Python
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
```