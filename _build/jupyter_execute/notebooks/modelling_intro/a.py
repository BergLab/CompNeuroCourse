#!/usr/bin/env python
# coding: utf-8

# # What is computational neuroscience?
# 
# The question is similar to asking "What is neuroscience?". This question can be answered as something like, "the scientific study of the nervous system"  [Wikipedia](https://en.wikipedia.org/wiki/Neuroscience). However, neuroscience is many things. It is multidisciplinary with many branches. Similarly, computational neuroscience is multidisciplinary. It is a subset of branches within neuroscience. Wikipedia defines it in the following way: 
# 
# 
# 
# 

# Wikipedia: [Computational Neuroscience](https://en.wikipedia.org/wiki/Computational_neuroscience):
# 
#     Computational neuroscience (also known as theoretical neuroscience or mathematical neuroscience) is a branch of neuroscience which employs mathematical models, theoretical analysis and abstractions of the brain to understand the principles that govern the development, structure, physiology and cognitive abilities of the nervous system.
#     
#     Computational neuroscience employs computational simulations to validate and solve mathematical models, and so can be seen as a sub-field of theoretical neuroscience; however, the two fields are often synonymous. The term mathematical neuroscience is also used sometimes, to stress the quantitative nature of the field.

# Goal is to achieve and abstraction and understanding of neural systems: Ie. Develop a theory and understanding
# Another aspect: quantitative data analysis often hand in hand with computational neuroscience. 

# ## A scientific theory?
# The word "theory" has more than one meaning. How do we define A theory? First, it can mean a supposition or a system of ideas intended to explain something, especially one based on general principles independent of the thing to be explained. Examples are Darwin's theory of evolution ([Darwinism](https://en.wikipedia.org/wiki/Darwinism)) or Newton's [Theory of gravity](https://en.wikipedia.org/wiki/Gravity). Second, a set of principles on which the practice of an activity is based: "a theory of education" and "music theory". Third, an idea used to account for a situation or justify a course of action: "my theory would be that the place has been seriously mismanaged". The latter is closest what is often associated with a scientific theory. It can explain an observation and it can be rejected if it is wrong.

# ### A hypothesis: A theory should contain a set of hypothesis (or generate hypotheses)
# A proposed explanation made on the basis of limited evidence as a starting point for further investigation. In mathematics we prove theorems, in science we reject hypothesis. 
# We can have hypothesis without a theory, but not vice versa.
# 

# ### Theory and models
# #### A model is an implementation of a theory. If can also be a realization of an aspect of the theory.
# 
# <div align="center">
# <p style="text-align:center;"><img src="/Figures/theory_chairs" width="600"/>
# </div>
# 
# <div align="center">
# <p style="text-align:center;"><img src="theory_chairs.png" width="800"/>
# </div>
# 
# A realisation of a theory in the shape of a model is important, because it can test the validity of both the model and the theory. An experiment can be conducted to verify the model and compare with another realisation. Hence, experiments or a computer simulation can reject one model and favor another model. This will help refining the theory (red arrows, figure ?)
# 
# <div align="center">
# <p style="text-align:center;"><img src="theory_models.png" width="400"/>
# </div>
# 

# ### Why is modelling important?
# * Aid in reasoning. Some phenomena can only be found when doing modeling. Especially in systems with many interacting parts (complex systems) and experiments only provide partial information.
# 
# * Remove ambiguities - sharpening the theory - intuition may turn out to be wrong. Also a theory may mean different things to different people. A good theory should have as little interpretable material as possible.
# 
# * Because experimental data is hard to get- models can replace it to some extent - and help develop a full understanding of a system. A good example is multi-compartmental models of neurons, which are now considered facts.
# 
# * Computer power allows to quickly seek answers and effectively extend the range of experiments
# 
# * Models can save time- rather than doing mindless experiments- especially important in the industry. Cohort size, medical dose, etc.
# 
# * Inspiration for new experiments. Something doesn’t make sense, we need to develop a new theory/understanding. This is the essence of science!

# Doing modelling also help us understand the often more complicated systems. It will help us move towards the right in the Dunning-Krueger plot. The Dunninger-Krueger effect is a hypothetical cognitive bias stating that people with low ability at a certain task often over-estimate their ability.
# 
# 
# <div align="center">
# <p style="text-align:center;"><img src="Dunning_Kruger.png" width="600"/>
# </div>
# 

# ### Chosing and shaping a model
# Some of the issues to consider when chosing is the following:
# * What type of model?
# * What part of the system?
# * How to deal with the parameters, which are not known from experiments?
# * What assumptions are “reasonable”? What is biologically plausible?

# ## Scale of a system
# The biological processes that make up an organism occur on various scales. Often a process on of a certain scale can be understood independently of the process on higher of lower scales. This could for instance be the molecular machinery of synapses. The fusion of the vesicle and the synaptic exocytose is a complicated process. Nevertheless, understanding the rich complexity of this process is not necessary for understanding how neuronal microcircuits work, for instance. Modelling often only works on one scale and it is difficult to connect models on one scale with a different. This is a challenge in neuroscience. 
# 
# <div align="right">
# <p style="text-align:center;"><img src="Scales1.png" width="200"/>
# </div>
# 
# 

# A clear challenge in neuroscience is a comprehensive mapping the connections between neurons. The research field of mapping the connections is called "connectomics". The challenge consists of one one hand it is required to image on a microscopic scale that is fine enought to confirm that there is a synaptic contact between two cells. On the other hand, the axons can travel over distances, which are many orders of magnitude hight than the scale of synaptic contacts. 
# 
# 
# <div align="right">
# <p style="text-align:center;"><img src="Scales2.png" width="200"/>
# </div>
# 
# 
# Wikipedia: [Connectomics](https://en.wikipedia.org/wiki/Connectomics):
# 
#     Connectomics is the production and study of connectomes: comprehensive maps of connections within an organism's nervous system
# 

# An example of the mixture of scales, which is required in mapping connections can be found in a recent study using rhesus monkey ([Xu et al 2021]( https://doi.org/10.1038/s41587-021-00986-5 )).  
# <video width="320" height="240" controls>
#   <source src="Scalesnervoussystem.mp4" type=video/mp4>
# </video>
# 
# 

# ## Bridging scales has a caveat: Reductionism
# There is a prevalent but simplistic belief in neuroscience that behavior of the organism can be explained or ascribed to a certain cell type. "Cell type x is responsible for behavior y". Behavior could even be ascribed to a certain ion channel protein or gene, which has been investigated by instance come from genetic knock-out. This of course a monumental simplification, since it would require explaining a phenomenon (behavior) which is the outcome of often the highest scale in the system (nervous system). To read more about these caveat see [Krakauer]( ) 
# 
# 
# <p style="text-align:left;"><img src="Reductionism.png" width="300"/> <p style="text-align:left;"><img src="Scales3.png" width="200"/>
# 
