"""
Script to generate PSNR graph
"""
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import rainbow
import numpy as np

# Load data
data = pd.read_csv('../psnr_results/psnr_compression_results.csv')

# Generate color map
colormap = rainbow(np.linspace(0, 1, 8))
fig = plt.figure()

# Select fista data
fista_none = data[(data['Selection function']=='fista') & (data['Test Stride Steps']==1) & (data['Normalized Dictionary'] == 'No') & (data['Normalized X'] == 'No')][['PSNR', 'Compression', 'Precision']]
fista_X = data[(data['Selection function']=='fista') & (data['Test Stride Steps']==1) & (data['Normalized Dictionary'] == 'No') & (data['Normalized X'] == 'Yes')][['PSNR', 'Compression', 'Precision']]
fista_D = data[(data['Selection function']=='fista') & (data['Test Stride Steps']==1) & (data['Normalized Dictionary'] == 'Yes') & (data['Normalized X'] == 'No')][['PSNR', 'Compression', 'Precision']]
fista_both = data[(data['Selection function']=='fista') & (data['Test Stride Steps']==1) & (data['Normalized Dictionary'] == 'Yes') & (data['Normalized X'] == 'Yes')][['PSNR', 'Compression', 'Precision']]




# Select vanilla data
vanilla_stride1 = data[(data['Selection function']=='vanilla') & (data['Test Stride Steps']==1)][['PSNR', 'Compression']]
vanilla_stride2 = data[(data['Selection function']=='vanilla') & (data['Test Stride Steps']==2)][['PSNR', 'Compression']]

# Select jpg data
jpg = data[data['Selection function']=='jpg'][['PSNR', 'Compression']]


# Plot jpg data and set ax object
ax = jpg.plot.scatter(x='Compression', y='PSNR', marker='*', c='m')

# Plot vanilla data
vanilla_stride1.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='o', c='r')
vanilla_stride2.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='o', c='g')

# Plot fista data
fista_none.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='k')
fista_X.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='y')
fista_D.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='c')
fista_both.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='m')

# Set Figure parametes
plt.xscale('log')
plt.grid(True)
plt.title('Compression-PSNR w.r.t. normalization schemes')
plt.legend(['JPG', 'Vanilla stride 1', 'Vanilla stride 2', 'FISTA stride 1, No normalization', 'FISTA stride 1, X normalization', 'FISTA stride 1, D normalization', 'FISTA stride 1, Both normalization'])
# plt.legend(['FISTA stride 1, fp8', 'FISTA stride 1, fp32', 'FISTA stride 2'])
plt.savefig('normalization_view_psnr_compression_results.png')


