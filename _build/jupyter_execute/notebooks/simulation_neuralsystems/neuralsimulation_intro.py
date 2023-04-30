#!/usr/bin/env python
# coding: utf-8

# # Simulating neural populations

# In this introduction chapter, we will delve into the fascinating field of neural simulation, focusing on simulating populations of spiking neural networks using the [Brian2 library](https://briansimulator.org/). Neural simulation is a powerful tool for investigating the dynamics and properties of biological neural systems, enabling researchers to model and analyze their behavior under various conditions. The Brian2 library is a versatile and user-friendly Python-based platform specifically designed for simulating spiking neural networks. It efficiently handles the underlying differential equations, automatically solving them and relieving us from the manual calculations we performed in the previous chapter. This allows for accurate simulations of large-scale networks with diverse neuron types and synaptic connections. Through hands-on examples and practical exercises, we will deepen our understanding of spiking neural network behavior, explore the effects of different network parameters, and learn how to create customized models that can provide insights into the underlying principles of biological neural systems. 

# ## Supporting material
# 
# This chapter's supporting material is [Neuromatch Academy playlist on biological neuron models](https://www.youtube.com/playlist?list=PLkBQOLLbi18MCEdPJQ7gdnqP-Z0Tkmcjy) (skip Q&As videos) and the following chapters from Sterratt's Principles of Computational Modelling in Neuroscience: 2.1, 2.2, 7.1, 7.2, 8.2 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo("MAOOPv3whZ0", width=600, height=400)

