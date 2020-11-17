"""
Script to generate PSNR graph
"""
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import rainbow
import numpy as np

# Load data
data = pd.read_csv('../psnr_results/psnr_decompression_data.csv')

# Generate color map
colormap = rainbow(np.linspace(0, 1, 8))
fig = plt.figure()

# Select fista data
fista_stride1 = data[(data['Selection function']=='fista') & (data['Test Stride Steps']=='1')][['PSNR', 'Compression', 'Precision']]
fista_stride1_f32 = fista_stride1[fista_stride1['Precision']=='32']
fista_stride1_f8 = fista_stride1[fista_stride1['Precision']=='8']
fista_stride2 = data[(data['Selection function']=='fista') & (data['Test Stride Steps']=='2')][['PSNR', 'Compression']]

# Select vanilla data
vanilla_stride1 = data[(data['Selection function']=='Vanilla') & (data['Test Stride Steps']=='1')][['PSNR', 'Compression']]
vanilla_stride2 = data[(data['Selection function']=='Vanilla') & (data['Test Stride Steps']=='2')][['PSNR', 'Compression']]
vanilla_stride3 = data[(data['Selection function']=='Vanilla') & (data['Test Stride Steps']=='3')][['PSNR', 'Compression']]

# Select jpg data
jpg = data[data['Selection function']=='jpg'][['PSNR', 'Compression']]


# Plot jpg data and set ax object
ax = jpg.plot.scatter(x='Compression', y='PSNR', marker='*', c='m')

# Plot vanilla data
vanilla_stride1.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='o', c='r')
vanilla_stride2.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='o', c='g')
vanilla_stride3.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='o', c='b')

# Plot fista data
fista_stride1_f8.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='k')
fista_stride1_f32.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='y')
fista_stride2.plot.scatter(ax=ax, x='Compression', y='PSNR', marker='s', c='c')

# Set Figure parametes
plt.xscale('log')
plt.grid(True)
plt.legend(['JPG', 'Vanilla stride 1', 'Vanilla stride 2', 'Vanilla stride 3', 'FISTA stride 1, fp8', 'FISTA stride 1, fp32', 'FISTA stride 2'])
# plt.legend(['FISTA stride 1, fp8', 'FISTA stride 1, fp32', 'FISTA stride 2'])
plt.savefig('full_psnr_decompression_results.png')


