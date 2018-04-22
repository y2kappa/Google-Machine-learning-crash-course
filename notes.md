#### notes.md

### Ch2. Linear regression

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

### Ch3. How do we reduce loss

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

### Ch4. First steps with Tensorflow

```python

import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)

```

Start tensorflow image in docker (see `https://www.tensorflow.org/install/install_mac`)

Run
```sh
docker run -it -p 8888:8888 tensorflow/tensorflow
```

Run all three exercises from `https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/programming-exercises`

Interesting points:
- removing outliers massively improves performance
- fine-tuning learning rate: increase if it keeps going down (that means it's too slow) or decrease if it goes randomly (it oversteps)
- graphically showing the regression line shows if it improves
- batch size is not clear yet, what is the relationship - a too big one seems to completely interfere with learning
- scatter plotting predictions vs targets shows if it's skewed. if it leans to one place in one dimension it means it's not well distributed and that feature is badly fit for prediction

### Ch5. Generalization

Generalization is something we should not be afraid of. Trying to create a curly line that perfectly fits our training data can be hard to maintain in the appearance of new data.

If we are trying to overfit for a specific point, then the resulting model might fail for other new points. One way of looking at things is saying that: most of the captured stuff is well modelled. For the uncaptured stuff - either these features are not perfect descriptors or those are exceptions and we still don't know why - another law.

Generalization theory: ` https://en.wikipedia.org/wiki/Generalizability_theory `

> The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.

No need to worry about that _in theory_ because we can divide data into test and training data. If the model trained with training data performs well against the test data then it means we can generalise quite well and is not overfit.

To do that - we sample training data from the entire population and use just that. 

Critical assumptions for why this works:
- the data has an underlying distribution
- we are drawing independently and identically from this distribution (we are not biasing when we draw)
- the distribution is stationary (does not change within the data itself). **this might be violated due to the data nature**
- we are always drawing examples from partitions of the same distribution **this might be violated if distribution over time changes**

### Ch6. Training and test sets

