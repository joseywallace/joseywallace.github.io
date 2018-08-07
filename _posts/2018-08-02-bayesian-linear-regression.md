---
layout: post
title: Multivariable linear regression
date:   2018-08-02
excerpt: "An short easy introduction to PyMC3"
image: "/images/pymc3_logo.png"
--- 

## Multivariable linear regression
### Create the data 

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

**In [4]:**

{% highlight python %}
plt.scatter(X_train[:,0], y_train, c='r', label='X1')
plt.scatter(X_train[:,1], y_train, c='b', label='X2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
{% endhighlight %}

<span class="image fit"><img src="{{ "/images/output_3_0.png" | absolute_url }}" alt="" /></span>
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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


<!-- Table -->
<h2>Table</h2>

<h3>Default</h3>
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
