#### notes.md

### Linear regression

In this equation: $y' = b + w_1x_1~$ the terms mean:

```
y = predicted label
b = bias or w0 (intercept)
w1 = weight of feature 1
x1 = feature (known input)
```

In a complicated model we would use:

$y' = b + w_1x_1 + w_2x_2 + ... + w_nx_n$

Training a model simply means learning (determining) good values for all the weights and the bias from labeled examples such that the predicted values are as close as possible to the real values. To get the predicted values near the real values you need to ` minimize the difference between them ` i.e. `minimize loss` 

In mathematical terms: the objective of the training model is $argmin(Loss function)$

How to define a loss function:
* Mean squared error:  
```math #yourmathlabel
MSE = 1/n \sum_{i=1} ^n (y^{hat}_i - y_i)^2
```
* There are others, this is only the most famous.

### How do we reduce loss

One way - 
1. Initialise your weights with random values
2. Find out the direction in which you move in the parameter space that keeps reducing the loss - `gradient descend`

This is easy if your problem is `convex` - you keep descending til you hit the minimum. 

However, this is not the case for neural nets. There's more than one minimum and the values you find truly depend on the initial values of your weights from step [1].

How to do in practice:
1. Find the direction (gradient) of your loss function w.r.t. loss function. How you calculate gradient - on one example, on all, on a batch? See below.
2. Move a step in your parameters in that direction - how far/fast, that depends on your `learning rate`. This determines how many steps you take and also if you overstep. There are alternatives to this: stochastic gradient descent and mini-batch stochastic gradient descent. 
3. Repeat until your loss stops decreasing.

> In gradient descent, a batch is the total number of examples you use to calculate the gradient in a single iteration. A large data set with randomly sampled examples probably contains redundant data.  In fact, redundancy becomes more likely as the batch size grows. Some redundancy can be useful to smooth out noisy gradients, but enormous batches tend not to carry much more predictive value than large batches.