---
layout: post
title: Bayesian multivariable linear regression in PyMC3
date:   2018-08-02
excerpt: "A motivating example for Sklearn users interested Bayesian analysis"
image: "/images/pymc3_logo.png"
--- 

Bayesian linear regression (BLR) is a powerful tool in the field of data science. For example, such models can provide a probability density of parameter values as opposed to a single best-fit value as in the standard (Frequentist) linear regression. In addition, BLR can be used to fit to parameters within a specified interval or create hierarchical models. However, despite decades of development and open source libraries, BLR has yet to reach its full user-base potential. One key obstacle to this is overcoming the barrier to entry for new users. In this blog post I hope to remove some of these obstacles by demonstrating how to;
	<!-- Lists -->
		<ol>
			<li>create a model </li>
			<li>train a model </li>
			<li>create a traceplot and summary statistics </li>
			<li>run the model on test data </li>
		</ol>
		

### Create the data 
In this first step, the necessary libraries are imported and the data set is created. PyMC3 is the Bayesian analysis library and tehano is the back-end of PyMC3. Theano is necessary to import in order to create shared variables that can be used to switch out the test and train data.

In this example, the data set has only two features and 1000 data points. However, these values can be changed through the *num_features* and *data_points* attributes. The variables *beta_set* and *alpha_set* are the slopes and intercept, respectively, that we will try to guess later.

The variables X and y are created using the slope and intercept values and normally distributed random noise is added to Y. Finally, X and y are split into training and testing set via Sklearn's train_test_split function.

**In [1]:**

{% highlight python %}
# create the data set
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
plt.style.use('seaborn-darkgrid')

num_features = 2
data_points = 1000
beta_set = 2.*np.random.normal(size=num_features)
alpha_set = np.random.normal(size=1)

X = np.random.normal(size=(data_points, num_features))
y = alpha_set + np.sum(beta_set*X, axis=1) + np.random.normal(size=(data_points))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
{% endhighlight %}

    WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
    
 
### Plot the data set 
Next, let's visualize the data we are fitting. 

**In [4]:**

{% highlight python %}
plt.scatter(X_train[:,0], y_train, c='r', label='X1')
plt.scatter(X_train[:,1], y_train, c='b', label='X2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
{% endhighlight %}

<span class="image fit">
	<img src="{{ "/images/output_3_0.png" | absolute_url }}" alt=""/>
</span>

### Create the pymc3 model 

**In [4]:**

{% highlight python %}
lin_reg_model = pm.Model()  #instantiate the pymc3 model object

#Create a shared theano variable. This allows the model to be created using the X_train/y_train data 
# and then test with the X_test/y_test data
model_input = theano.shared(X_train)
model_output = theano.shared(y_train)

with lin_reg_model:
    alpha = pm.Normal('alpha', mu=0, sd=1, shape= (1)) # create random variable alpha of shape 1
    beta = pm.Normal('betas', mu=0, sd=1, shape=(1,num_features)) # create random variable beta of shape 1,num_features
    
    s = pm.HalfNormal('s', sd=1)  # create distribution to describe noise in the data
    
    data_est = alpha + theano.tensor.dot(beta, model_input.T)  # Expected value of outcome
    
    y = pm.Normal('y', mu=data_est, sd=s, observed=y_train)  # Likelihood (sampling distribution) of observations
{% endhighlight %}
 
### Train the model 

**In [5]:**

{% highlight python %}
with lin_reg_model:
    step = pm.NUTS()
    nuts_trace = pm.sample(2000, step, njobs=1)
{% endhighlight %}

    Sequential sampling (2 chains in 1 job)
    NUTS: [s, betas, alpha]
    100%|█████████████████████████████████████████████████████████████████████████████| 2500/2500 [00:05<00:00, 432.98it/s]
    100%|█████████████████████████████████████████████████████████████████████████████| 2500/2500 [00:02<00:00, 836.41it/s]
    
 
### traceplot and summary statistics 

**In [6]:**

{% highlight python %}
pm.traceplot(nuts_trace)
plt.show()
{% endhighlight %}

 
<span class="image fit"><img src="{{ "/images/output_9_0.png" | absolute_url }}" alt="" /></span>

**In [7]:**

{% highlight python %}
pm.summary(nuts_trace[1000:])
{% endhighlight %}




<div class="table-wrapper">
	<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha__0</th>
      <td>0.480106</td>
      <td>0.037595</td>
      <td>0.000774</td>
      <td>0.405709</td>
      <td>0.550487</td>
      <td>2975.210769</td>
      <td>1.001104</td>
    </tr>
    <tr>
      <th>betas__0_0</th>
      <td>1.557257</td>
      <td>0.036890</td>
      <td>0.000658</td>
      <td>1.487401</td>
      <td>1.628156</td>
      <td>3147.314093</td>
      <td>0.999515</td>
    </tr>
    <tr>
      <th>betas__0_1</th>
      <td>-1.562830</td>
      <td>0.038143</td>
      <td>0.000744</td>
      <td>-1.634971</td>
      <td>-1.484338</td>
      <td>2849.492200</td>
      <td>0.999755</td>
    </tr>
    <tr>
      <th>s</th>
      <td>0.986880</td>
      <td>0.027428</td>
      <td>0.000468</td>
      <td>0.930528</td>
      <td>1.038020</td>
      <td>3049.781251</td>
      <td>0.999667</td>
    </tr>
  </tbody>
</table>
</div>

 
### Test the trained model on the test data 

**In [8]:**

{% highlight python %}
# set the shared values to the test data
model_input.set_value(X_test)
model_output.set_value(y_test)

#create a posterior predictive check (ppc) to sample the space of test x values
ppc = pm.sample_ppc(
        nuts_trace[1000:], # specify the trace and exclude the first 1000 samples 
        model=lin_reg_model, # specify the trained model
        samples=1000) #for each point in X_test, create 1000 samples
{% endhighlight %}

    100%|████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 1683.52it/s]
    
 
### Plot the y_test vs y_predict and calculate r**2 

**In [9]:**

{% highlight python %}
pred = ppc['y'].mean(axis=0)  # take the mean value of the 1000 samples at each X_test value 
plt.scatter(y_test, pred)
plt.show()
r2_score(y_test,*pred)
{% endhighlight %}

 
<span class="image fit"><img src="{{ "/images/output_14_0.png" | absolute_url }}" alt="" /></span>




    0.835496545799272
