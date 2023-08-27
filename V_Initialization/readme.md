**Initialization:**

- **Different initializations lead to different results**
- The weights $W^{[l]}$ **should be initialized randomly** to break symmetry. Then each neuron would learn a different function of its inputs.
- **Don't initialize to values that are too large.**
- When initializing weights with large random values results, the the last activation function (sigmoid) outputs results close to 0 or 1. This leads to a very high loss for that example at the start of training. Optimization algorithm can be eventually slowed down by using large random numbers.
- **He initialization works well for networks with ReLU activations.**
- It is however okay to initialize the biases $b^{[l]}$ to zeros. Symmetry is still broken so long as $W^{[l]}$ is initialized randomly.