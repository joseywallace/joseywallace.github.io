---
layout: post
title: Dynamic programming and Temporal difference learning from scratch
date:   2020-04-06
excerpt: "Implementation of the three foundational dynamic programming and temporal difference learning methods in python"
image: "/images/2020-04-15-logo.png"
--- 

In this blog post we apply value iteration, policy iteration, and q tabulation to solve the frozen lake problem. Value and policy iteration are dynamic programming algorithms that can be used to determine the optimal policy given a perfect model of the environment. Such algorithms are of limited practical utility in Reinforcement Learning (RL) due to their restriction of requiring a perfect model of the environment. However, many of the fundamental RL algorithms are based on these DP algorithms. The third algorithm implemented in this post is the well-known off-policy temporal difference learning algorithm known as Q-learning. In contrast with the DP algorithms, Q-learning doesn't require a model of the environment and hence can be trained directly from raw experience (similar to Monte Carlo methods). However, like the DP algorithms, Q-learning updates its estimates on the fly without waiting to see the final outcome. 

### Creating the Environment
In this section we create the forzen lake environement. The frozen lake environment is meant to model a robot trying to quickly cross a frozen lake. There are multiple holes in the ice, which the robot will fall into if it tries to pass over. The lake is modeled by a numpy array (shown below) where the zeros represents solid ice and ones represents a hole in the ice. The robot starts at the upper right corner (0,0) and must find its way to the lower left corner (5,5). There are four available actions; up, right, left, and down. At each step, there is an 80% chance the robot will go the intended direction and a 20% chance that the robot will go either left or right of the intended direction. Each step incurs a -0.1 reward. Falling into the ice ends the episode with a -1 reward. Reaching the goal ends the episode with a +1 reward. 

{% highlight python %}
import numpy as np
import random
SMALL_LAKE = np.array([[0,0,0,0,1],
                       [0,0,1,1,0],
                       [1,0,0,0,0],
                       [0,0,1,0,1],
                       [0,0,0,0,0]])
STEP_COST = -0.1
class FrozenLake():
    def __init__(self):
        self.lake = SMALL_LAKE
        self.goal = np.array([d-1 for d in self.lake.shape])
        self.action_space = np.array([0,1,2,3]) #up, right, down, left
        self.translation = [[-1,0],[0,1],[1,0],[0,-1]]
        self.state_space = list(map(tuple, np.indices(self.lake.shape).T.reshape(-1,2)))
    def reset(self):
        self.state = np.array([0,0])
        return tuple(self.state)
    def step(self, action):
        assert(action in self.action_space), "action "+str(action)+" not in action space"
        reward = STEP_COST
        rnd = np.random.rand()
        if rnd < 0.1: action = (action+1)%self.action_space.size
        elif rnd < 0.2: action = (action-1)%self.action_space.size
        new_state = (self.state + self.translation[action])
        if all((new_state >= np.array([0,0])) & (new_state < self.lake.shape)):
            self.state = new_state
        if self.lake[tuple(self.state)] == 1: # fall into hole
            return tuple(self.state), reward-1, True
        elif all(self.state == self.goal): # goal reached
            return tuple(self.state), reward+1, True
        else:
            return tuple(self.state), reward, False
    def get_state(self):
        return tuple(self.state)
{% endhighlight %}

The two main functions of the environment that the agent will use are the *reset* and *step* functions. The reset function moves the robot back to the start position and the step function accepts an action and returns the resulting state, reward, and boolean describing if the episode has ended.  

### Policy Iteration

### Value Iteration

As mentioned above, value iteration is a DP algorithm for determining the optimal policy given a perfect model of the environment. The algorithm 

In this post I go through the mathematics behind neural networks and demonstrate how to implement a neural network in python from scratch. First, I'll introduce the simple example of a single neuron with a sigmoid activation function. The neuron is the fundamental unit from which neural networks are designed. The single neuron model is used to introduce the concepts of gradient descent and back propagation before applying these concepts to the more complicated neural network.

The perceptron, which was the precursor to the sigmoid neuron was invented by Frank Rosenblatt in the 1950s and 60s. The concept of the perceptron, which is biologically inspired, takes the sum of multiple weighted inputs and compares this sum to some threshold value. If the value is greater than the threshold, the neuron 'fires' creating an output of unity. Otherwise, the neuron outputs zero. Neurons can be programmed by adjusting their weights and biases, to simulate 'and', 'or', and 'nand' gates. Multiple neurons can be strung together to form literally any logic circuit imaginable. If we can then string multiple of these neurons together in layers and somehow optimize the system towards a particular solution to a problem, then we have an algorithm for designing complex possibly non-intuitive circuits that given some input can produce the desired output. That’s a huge if.. It's truly amazing that such a technique exist. The two key components that make up this optimization technique are back propagation and stochastic gradient descent. Both are actually fairly simple to understand requiring only an understanding of basic calculus. Gradient defines the method by which we seek to minimize the cost function and hence the error in our model. Back propagation is a technique for optimizing the influence of a each weight and biases on the cost function.

The figure below shows the simple single neuron model with two inputs and a single output.

<figure>
<span class="image fit">
        <img src="{{ "/images/node.png" | absolute_url }}" alt=""/>
        <figcaption><b>Figure 1.</b> Single neuron model with two inputs.</figcaption>
</span>
</figure>

It is useful to define *z* as shown below:

<figure>
<span class="image fit">
        <img src="{{ "/images/eq01.png" | absolute_url }}" alt=""/>
</span>
</figure>


Where *w<sub>k</sub>* are the weights, *x<sub>k</sub>* are the input values, and *b* is the bias. The *z* function, known as the weighted input to the neuron, is passed through a non-linearity function in this example a sigmoid. The sigmoid function allows us to optimize the weights and biases much more easily since as opposed to the perceptron model discussed above, the sigmoid is continuous. In addition the derivative of the sigmoid is simply &#963; = &#963;(1-&#963;). The final output of the neuron, *a*, is shown below along with the &#963; function.

<figure>
<span class="image fit">
        <img src="{{ "/images/eq02.png" | absolute_url }}" alt=""/>
</span>
</figure>

The cost function can be defined as the square of the difference between *a* and the true value *y* as shown below:

<figure>
<span class="image fit">
        <img src="{{ "/images/eq03.png" | absolute_url }}" alt=""/>
</span>
</figure>


As discussed above, our approach to optimize this neuron is to find the gradient of the cost function with respect to the weights and bias and step in the opposite direction of this gradient. The gradient is derived via the chain rule as shown below:

<figure>
<span class="image fit">
        <img src="{{ "/images/eq04.png" | absolute_url }}" alt=""/>
</span>
</figure>

Once we have found the gradient, we want to step in the opposite direction of the gradient by some step size which is defined by the learning rate &#951;.

Let's put this all together in python now. The Neuron class shown below is initialized by passing in the number of epochs (training steps over the entire dataset), eta (learning rate), and dim (length of input vector X).

{% highlight python %}
import numpy as np
class Neuron:
    def __init__(self, dim, epochs, eta):
        self.epochs = epochs
        self.eta = eta
        self.w = np.array([0.1]*dim)
    def fit(self, x, y):
        for i in range(self.epochs):
            del_w =  del_b = 0
            for xt, yt in zip(x,y):
                z = self.w[0] + self.w[1:].dot(xt)
                sigz = self.sigma(z)
                delC = 2*(yt - sigz)*sigz*(1-sigz)
                del_w += delC*xt
                del_b += delC
            self.w[0] += (self.eta/len(x))*del_b
            self.w[1:] += (self.eta/len(x))*del_w
    def sigma(self, z):
        return 1/(1+np.exp(-z))
{% endhighlight %}

The single neuron acts as a linear separator and hence we can test the Neuron class on the data set created below:

{% highlight python %}
x = 2*np.random.rand(100,2) - 1
y = x.dot((1,-1.5)) > 0.4

n = Neuron(2, 1500, 1)
n.fit(x,y)

plt.scatter(x[:,0], x[:,1], c=y)
z = x.dot(n.w[1:]) + w[0]
plt.scatter(x[:,0], x[:,1], c=n.sigma(z))
{% endhighlight %}

<figure>
<span class="image fit">
        <img src="{{ "/images/neuron_data.png" | absolute_url }}" alt=""/>
        <figcaption><b>Figure 2.</b> (left) raw data color coded by label and (right) predictions made by the single neuron model after training.</figcaption>
</span>
</figure>

1. Partition the data into minibatches
2. Calculate the gradient of the cost function via back propagation
3. Calculate the step based on learning rate and gradient
4. Apply a step change to weights and biases
5. Rinse and repeat for each minibatch for n epochs
 

We can break this into two functions (1) stochastic gradient descent (sgd) and (2) backpropagation. The sgd method is responsible for all of the above steps except 3, which the back propagation method handles. The code for the sgd method is shown below. The method loops over though the data *epochs* number of times, shuffling the data and creating new minibatches at each epoch. Then, the code loops through each minibatch and each instance within each minibatch creating three nested for loops. The runtime is *O(epochs\*len(x))*.

{% highlight python %}
    def sgd(self, x, y, epochs, eta, batch_size, x_test=None, y_test=None):
        for j in range(epochs):
            idx = np.random.permutation(x.shape[0])
            xr, yr = x[idx], y[idx]
            minibatches = [(xr[k:k+batch_size], yr[k:k+batch_size])
                        for k in range(0, xr.shape[0], batch_size)]
            for xm, ym in minibatches:
                grad_w = [np.zeros(w.shape) for w in self.w]
                grad_b = [np.zeros(b.shape) for b in self.b]
                for xi, yi in zip(xm, ym):
                    d_grad_w, d_grad_b = self.backpropagate(xi, yi)
                    grad_w = [dw+w for dw, w in zip(d_grad_w, grad_w)]
                    grad_b = [db+b for db, b in zip(d_grad_b, grad_b)]
                self.w = [w-eta*(dw/xm.shape[0]) for w, dw in zip(self.w, grad_w)]
                self.b = [b-eta*(db/xm.shape[0]) for b, db in zip(self.b, grad_b)]
            if x_test and y_test:
                print('Epoch:', j, 'Score:', self.score(x_test, y_test), '%')
{% endhighlight %}

The next function is backpropagation. This function first takes in a single training instance and returns the weight and bias gradients for that instance. First, the function performs a forward pass through the network, collecting the *z* and *a* values along the way. Then, the &#948;, *w*, and *b* are calculated for the last layer in the network (this layer is calculated differently than the rest of the layers due to the cost function). Next, the &#948;, *w*, and *b* for each subsequent layer working back towards the input of the neural network are calculated using an iterative dynamic programming approach. Finally, the resulting gradient of *w* and *b* is returned. 

{% highlight python %}
    def backpropagate(self, x, y):
        grad_w = [np.zeros(w.shape) for w in self.w]
        grad_b = [np.zeros(b.shape) for b in self.b]
        #forward pass to calculate the 'z' and 'a' values
        a_vals, z_vals = [x], []
        for w, b in zip(self.w, self.b):
            z_vals += [np.matmul(w, a_vals[-1]) + b]
            a_vals += [sigmoid(z_vals[-1])]
        #backward pass propagating errors through the network
        delta = (a_vals[-1] - y)*sig_prime(z_vals[-1])
        grad_w[-1] = delta.reshape(-1,1)*a_vals[-2]
        grad_b[-1] = delta
        for l in range(2, self.n):
            delta = np.matmul(self.w[-l+1].T, delta)*sig_prime(z_vals[-l])
            grad_w[-l] = delta.reshape(-1,1)*a_vals[-l-1]
            grad_b[-l] = delta
        return grad_w, grad_b 
{% endhighlight %}

As a quick validation, we can test a neural networks ability to fit non-linearly-separable data. Figure 5 below shows the result of fitting a neural network with sizes = (2,3,2) to a data set where all labels within a given radius are classified as 1 and all outside the radius are classified as 0. 

<figure>
<span class="image fit">
        <img src="{{ "/images/mlp_data.png" | absolute_url }}" alt=""/>
        <figcaption><b>Figure 4.</b> (left) Raw data created to test the neural network and (right) results of training a neural network with layer sizes (2,3,2) on the left data. The neural network performs decently well only missing a few instances. </figcaption>
</span>
</figure>

### Resources and Additional Reading

[1] <a href ="http://neuralnetworksanddeeplearning.com/">Neural networks and Deep Learning by Michael Nielsen, Dec. 2019</a>

[2] <a href ="https://www.youtube.com/watch?v=aircAruvnKk&t=832s">3Blue1Brown youtube series on neural networks</a>

[3] <a href ="https://github.com/joseywallace?tab=repositories">Code repository for this blog post</a>
