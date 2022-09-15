#!/usr/bin/env python
# coding: utf-8

# # Introduction to Brian part 4: Build your own simulation
# 
# Now you are ready to build and simulate any neural population that you can conceive. 
# Your goal in this exercies is simply design, simulate and study a population of neurons.
# 
# Start by designing a population of neurons by choosing the following parameters:
# 
# ### Topics covered on week1a
# 
# - [ ] Number of neurons in the populations
# - [ ] Type of neurons: excitatory, inhibitory.. how many of each?
# - [ ] Connectivity: all connected? random connectivity?
# 
# Then implement it with Brian re-using section of code from the previous notebooks.
# 
# As before we start by importing the Brian package and setting up matplotlib for IPython:

# As always, we start by importing the functions from brian library:

# In[1]:


get_ipython().system('pip3 install brian2')
from brian2 import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# ## Interactive exploration
# 
# Colabs have a form feature which allows you to create interactive plots:
# (If you're using jupyer notebooks rather than Colab, you can use [Ipython widgets](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html))

# In[2]:


#@title Default title text { run: "auto", vertical-output: true }
populationSize = 1000 #@param {type:"slider", min:0, max:1000, step:1}

