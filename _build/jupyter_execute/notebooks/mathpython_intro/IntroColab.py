#!/usr/bin/env python
# coding: utf-8

# # Google's Colab
# 
# Have a look around. On the left side of the window you see there are four menu items:
#   - Table of content: shows the table of content of the notebook
#   - Search: let you search for worlds in the notebook
#   - Variables: keep tracks of evaluated variables, their type and values
#   - Files: the files present in the working directory
# 
# 

# ## Cells
# A notebook is a list of cells. Cells contain either explanatory text or executable code and its output. Click a cell to select it.

# ### Code cells
# Below is a **code cell**. Once the toolbar button indicates CONNECTED, click in the cell to select it and execute the contents in the following ways:
# 
# * Click the **Play icon** in the left gutter of the cell;
# * Type **Cmd/Ctrl+Enter** to run the cell in place;
# * Type **Shift+Enter** to run the cell and move focus to the next cell (adding one if none exists); or
# * Type **Alt+Enter** to run the cell and insert a new code cell immediately below it.
# 
# There are additional options for running some or all cells in the **Runtime** menu.
# 

# In[1]:


seconds_in_a_day = 24 * 60 * 60
print(seconds_in_a_day)


# ### Comments in code cells
# 
# In python we can add comments in our code by appending the symbol `#` before line. Comments are a great way of adding notes on the code for ourself or for someone we're sharing the code with (including our future self). Like python, Notebooks/Colab code cells also accept comments:

# In[2]:


# The variable ratsAlive stores the number of rats that survived the surgery
ratsAlive = 5


# With Colab you can harness the full power of popular Python libraries to analyze and visualize data. The code cell below uses **numpy** to generate some random data, and uses **matplotlib** to visualize it. To edit the code, just click the cell and start editing.

# In[3]:


import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()


# ### Text cells
# This is a **text cell**. You can **double-click** to edit this cell. Text cells
# use markdown syntax. To learn more, see our [markdown
# guide](/notebooks/markdown_guide.ipynb).
# 
# You can also add math to text cells using [LaTeX](http://www.latex-project.org/)
# to be rendered by [MathJax](https://www.mathjax.org). Just place the statement
# within a pair of **\$** signs. For example `$\sqrt{3x-1}+(1+x)^2$` becomes
# $\sqrt{3x-1}+(1+x)^2.$
# 

# ### Adding and moving cells
# You can add new cells by using the **+ CODE** and **+ TEXT** buttons that show when you hover between cells. These buttons are also in the toolbar above the notebook where they can be used to add a cell below the currently selected cell.
# 
# You can move a cell by selecting it and clicking **Cell Up** or **Cell Down** in the top toolbar. 
# 
# Consecutive cells can be selected by "lasso selection" by dragging from outside one cell and through the group.  Non-adjacent cells can be selected concurrently by clicking one and then holding down Ctrl while clicking another.  Similarly, using Shift instead of Ctrl will select all intermediate cells.

# ## Working with python
# Colaboratory is built on top of [Jupyter Notebook](https://jupyter.org/). Below are some examples of convenience functions provided.

# Long running python processes can be interrupted. Run the following cell and select **Runtime -> Interrupt execution** (*hotkey: Cmd/Ctrl-M I*) to stop execution.

# In[4]:


import time
print("Sleeping")
time.sleep(30) # sleep for a while; interrupt me!
print("Done Sleeping")


# ## Terminal commands
# 
# You can use the cells as you would normally use the terminal by simply adding the exclamation mark *!* before the command.
# 
# If you click on the left side menu item *files*, you'll see that there is a folder named *sampled_data*. Let's use the terminal command `cd` to enter the folder and the command `ls` to show the files present in the folder. The command `&&` allow use to link two terminal commands.
# 
# 
# 

# In[ ]:


get_ipython().system('cd sample_data && ls')


# ## Installing external libraries
# 
# Colabs come with a set of basic python libraries by default like *numpy* or *matplotlib*. However, other libraries are present by the default. 
# 
# For example, in week 3 we will Brian2 library to simulate spiking neural populations, let's try to import it:

# 

# In[ ]:


import brian2


# We get an error saying that there isn't any module named *brian2*. To use it, we first need to install it. If we want to use other libraries we can install them using `pip`. we can simply install it with the command `!pip install brian2`:

# In[ ]:


get_ipython().system('pip install brian2')


# Now let's try to import brian2 again:

# In[ ]:


import brian2


# ## Automatic completions and exploring code
# 
# Colab provides automatic completions to explore attributes of Python objects, as well as to quickly view documentation strings. As an example, first run the following cell to import the  [`numpy`](http://www.numpy.org) module.

# In[ ]:


import numpy as np


# If you now insert your cursor after `np` and press **Period**(`.`), you will see the list of available completions within the `np` module. Completions can be opened again by using **Ctrl+Space**.

# In[ ]:


np


# If you type an open parenthesis after any function or class in the module, you will see a pop-up of its documentation string:

# In[ ]:


np.ndarray


# The documentation can be opened again using **Ctrl+Shift+Space** or you can view the documentation for method by mouse hovering over the method name.
# 
# When hovering over the method name the `Open in tab` link will open the documentation in a persistent pane. The `View source` link will navigate to the source code for the method.

# ## Exception Formatting

# Exceptions are formatted nicely in Colab outputs:

# In[ ]:


x = 1
y = 4
z = y/(1-x)


# ## Rich, interactive outputs
# Until now all of the generated outputs have been text, but they can be more interesting, like the chart below. 

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Fills and Alpha Example")
plt.show()


# ## Integration with Drive
# 
# Colaboratory is integrated with Google Drive. It allows you to share, comment, and collaborate on the same document with multiple people:
# 
# * The **SHARE** button (top-right of the toolbar) allows you to share the notebook and control permissions set on it.
# 
# * **File->Make a Copy** creates a copy of the notebook in Drive.

# ## Loading Public Notebooks Directly from GitHub
# 
# Colab can load public github notebooks directly, this is a quick way of using other people notebooks!
# There are different ways of doing this:
# 
#     * Open Colab, click on file -> Open -> GitHub and search for the username and repo where the .ipynb you want to open is found
#     * Use the 'Open in Colab' chrome/firefox extension
#     * Manually copy the github path: [more info here](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
# 

# ## Commenting on a cell
# You can comment on a Colaboratory notebook like you would on a Google Document. Comments are attached to cells, and are displayed next to the cell they refer to. Comments are a nice tool to discuss some code with other people.
# 
# If you have edit or comment permissions you can comment by:
# 
# 1. Select a cell and click the comment button in the toolbar above the top-right corner of the cell.
# 1. Right click a text cell and select **Add a comment** from the context menu.
# 
# The Comment button in the top-right corner of the page shows all comments attached to the notebook. 

# ## Advance: The cell magics in IPython
# 
# IPython has a system of commands we call 'magics' that provide effectively a mini command language that is orthogonal to the syntax of Python and is extensible by the user with new commands. Magics are meant to be typed interactively, so they use command-line conventions, such as using whitespace for separating arguments, dashes for options and other conventions typical of a command-line environment.
# 
# Magics come in two kinds:
# 
#     Line magics: these are commands prepended by one % character and whose arguments only extend to the end of the current line.
#     Cell magics: these use two percent characters as a marker (%%), and they receive as argument both the current line where they are declared and the whole body of the cell. Note that cell magics can only be used as the first line in a cell, and as a general principle they can't be 'stacked' (i.e. you can only use one cell magic per cell). A few of them, because of how they operate, can be stacked, but that is something you will discover on a case by case basis.
# 
# The `%lsmagic` magic is used to list all available magics, and it will show both line and cell magics currently defined:

# In[ ]:


get_ipython().run_line_magic('lsmagic', '')


# An useful *magic* command is `%timeit` which allows us to compute how long a program takes to run:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

def computeEigenVals(matrix):
    return np.linalg.eigvals(matrix)

get_ipython().run_line_magic('timeit', 'computeEigenVals(np.random.rand(100,100))')

