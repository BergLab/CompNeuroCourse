#%%

df.plot.scatter('Temperature', 'Luminosity')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xscale('log')

#%%
df['volume'] = 4/3 * 3.14 * (df['Size'])**3
#%%
l = []
for startype in df['Type'].unique():
    if 'Dwarf' in startype:
        density = 10**5
    elif 'Giants' in startype:
        density = 10**-8
    elif 'Main' in startype:
        density = 1
    l.append(df[df['Type'] == startype]['Volume']*density)

df['Mass'] = pd.concat(l,axis=0)