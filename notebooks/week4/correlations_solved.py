#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
pd.options.mode.chained_assignment = None  # default='warn'

#%%
df = pd.read_csv('Stars.csv')
df.head()
#%%
df['Temperature']
#%%
np.corrcoef(df['Temperature'], df['Temperature'])
#%%
df.plot.scatter('Temperature','Temperature', figsize=(10,7))
r, p = stats.pearsonr(df['Temperature'], df['Temperature'])
print(f"The correlation coefficient is {r}")
#%%
df.plot.scatter('Temperature','Luminosity', figsize=(10,7))
r, p = stats.pearsonr(df['Luminosity'], df['Temperature'])
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 
#%%
r, p = stats.pearsonr(df['A_M'], df['Luminosity'])
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 
#%%
df.plot.scatter('Size','Luminosity', figsize=(10,7))
r, p = stats.pearsonr(df['Size'], df['Luminosity'])
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 
#%%
df.plot.scatter('Luminosity','A_M', figsize=(10,7))
r, p = stats.pearsonr(df['A_M'], df['Luminosity'])
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 
#%%
df.plot.scatter('Luminosity','A_M', figsize=(10,7))
r, p = stats.pearsonr(np.log(df['Luminosity']), df['A_M'])
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 
#%%
df.plot.scatter('Luminosity', 'A_M', figsize=(10,7))
r, p = stats.pearsonr(df['Luminosity'], df['A_M'])
# plt.yscale('log')
print(f"The correlation coefficient is {r}")
print(f"(double-tailed) p-value is {p}") 

#%%
df.corr()