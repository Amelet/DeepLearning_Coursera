# Python basics

## Why we use numpy library
The "math" library is designed for scalar mathematical operations, "numpy" is designed for vectorized operations on matrices and vectors, making it more suitable for deep learning tasks.

## Sigmoid gradient
```math
\sigma'(z) = \sigma(z) \times (1 - \sigma(z))
```
The gradient of the sigmoid function at any point `z` is given by the value of the sigmoid function at that point multiplied by the difference between 1 and the value of the sigmoid function at that point. This result is useful when implementing neural networks, especially during the backpropagation step.

## Unrolling arrays using np.reshape()
An image is represented by a 3D array of shape $(length, height, depth = 3)$. To become in input to an algorithm, such a 3D matrix can be unrolled into a 1D vector of shape $(length * height * 3, 1)$. In other words, you "unroll" the matrix.

## Normalizing data
Normalizing data often leads to a faster gradient descent convergence. 

## Softmax function and broadcasting in python
Softmax function:
```math
\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
```

*Outputs a Probability Distribution:* Each element of the output vector is in the range (0, 1), and the sum of the elements is 1.

*Exponential Amplification:* The exponential function amplifies the differences between the elements of the input vector. If one of the scores is slightly larger than the others, it will get a much larger share of the probability distribution.

*Used for Multi-class Classification:* It's especially useful when the neural network needs to decide among multiple classes. For binary classification, a sigmoid activation in the output layer is usually sufficient.

## Vectorization
Vectorized implementation is cleaner and faster.
```python
dot = np.dot(x1,x2)
outer = np.outer(x1,x2)
mul = np.multiply(x1,x2)

```

## L1 and L2 loss functions
### L1
The L1 loss function, also known as the "Least Absolute Deviations" (LAD) or "Mean Absolute Error" (MAE), calculates the sum of the absolute differences between the predicted values and the actual values. 
```math
L_1 = \sum_{i=1}^{n} |y_i - \hat{y}_i|
```

*Advantages:* It's more robust to outliers compared to the L2 loss because it doesn't square the differences.

*Drawbacks:* The gradients for L1 loss can be less consistent than for L2, as the derivative is a constant for non-zero errors.

### L2
The L2 loss function, also known as "Least Squares Error" (LSE) or "Mean Squared Error" (MSE), calculates the sum of the squared differences between the predicted and actual values.
```math
L_2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

```

*Advantages:* It's differentiable, making it more suitable for gradient-based optimization methods. The squared term also penalizes larger errors more than smaller ones, leading to a more stable and consistent learning process when the data has fewer outliers.

*Drawbacks:* It's sensitive to outliers because of the squared term.

In many contexts, a combination of L1 and L2 loss, called the Elastic Net, is used to leverage the strengths of both loss functions.
