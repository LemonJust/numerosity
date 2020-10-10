# %%
from IPython import get_ipython;

get_ipython().magic('reset -sf')

# %% imports
import os
import numpy as np
import pandas as pd
import more_itertools as it

import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter
from sklearn.decomposition import PCA


# %%

#  custom functions
def get_baseline(signal, window, percentile):
    baseline = []

    for window_data in it.windowed(signal.T, n=window, step=1):
        base = np.percentile(np.array(window_data), percentile, axis=0)
        baseline.append(base)

    return np.array(baseline).T


def crop_to_window(signal, window):
    """
    Returns signal without half window at the begining and half window at the end.
    Crops along 0 dimention!!!!

    :param signal: signal to crop
    :param window: sliidng window size
    :return: cropped signal
    """
    half_window = np.floor(window / 2).astype(int)
    crop_sig = signal[half_window:-half_window]

    return crop_sig


def get_dff(signal, baseline=None):
    if baseline is None:
        window = 21
        percentile = 8
        baseline = get_baseline(signal, window, percentile)
        signal = crop_to_window(signal.T, window).T

    return (signal - baseline) / baseline


def get_stim(stim_pattern, stim_len):
    """
    Returns np.array of int :  stimuli label for the current state.
    Assuming b1>1>b2>3

    :param stim_pattern: the pattern to repeat
    :param stim_len: stimulus length
    :return: array of indices : 0 for b1 , 1 for 1 dot, 2 for b2 and 3 to 3 points.
    """
    stim_cycle = len(stim_pattern)

    # can modify here to start at any point
    stimulus = (stim_pattern * int(np.ceil(stim_len / stim_cycle)))[0:stim_len]
    stimulus = [int(x) for x in stimulus]
    return np.array(stimulus)


# %% load extracted signals
input_folder = 'D:/Code/repos/numerosity/Anna_20200717/all_neurons/fish1'
fish_name = 'fish1'
input_file = f'{input_folder}/{fish_name}_signals_diam2_NxT.csv'
signal = np.loadtxt(input_file, delimiter=',', dtype=float)

print(f'N cells : {len(signal)}')
X = signal
# %% Preprocess for PCA
dff = get_dff(signal)
print(f'dff shape : {dff.shape}')
X = dff.T
# %%
pca = PCA(n_components=100)
pca.fit(X)

plt.plot((pca.explained_variance_ / np.sum(pca.explained_variance_)) * 100)
plt.xlabel('PC')
plt.ylabel('% var. explained')
plt.xlim([0, 10])
plt.show()

# %%
X_pca = pca.transform(X)
# %%
plt.subplot(2, 2, 1)
plt.plot(X_pca[:, 0], X_pca[:, 1], '.')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.subplot(2, 2, 2)
plt.plot(X_pca[:, 0], X_pca[:, 2], '.')
plt.xlabel('PC 1')
plt.ylabel('PC 3')

plt.subplot(2, 2, 3)
plt.plot(X_pca[:, 1], X_pca[:, 2], '.')
plt.xlabel('PC 2')
plt.ylabel('PC 3')

plt.subplot(2, 2, 4)
plt.plot(X_pca[:, 1], X_pca[:, 3], '.')
plt.xlabel('PC 2')
plt.ylabel('PC 4')
plt.show()

# %%
PC1_neg = X_pca[0, :] < 0
PC1_pos = X_pca[0, :] > 0
print(f'num positive : {np.sum(PC1_pos)}, num negative : {np.sum(PC1_neg)}')
# %%
PC1_temporal = np.mean(X * X_pca[:, [0]], axis=0)
plt.plot(PC1_temporal)
plt.ylabel('I, a.u.')
plt.xlabel('time')
plt.title('PC1')
plt.show()
# %%
PC1_temporal = np.mean(X * X_pca[:, [1]], axis=0)
plt.plot(PC1_temporal)
plt.ylabel('I, a.u.')
plt.xlabel('time')
plt.title('PC2')
plt.xlim([750, 800])
plt.show()

# %%
# %% Generate stimulus pattern
window = 21
percentile = 8
stim_pattern = '00000111112222233333'
stimulus = get_stim(stim_pattern, signal.shape[1])
crop_stim = crop_to_window(stimulus, window)
# %% plot box-plots for DFF

blank1 = dff[:, crop_stim == 0]
dot1 = dff[:, crop_stim == 1]
blank2 = dff[:, crop_stim == 2]
dot3 = dff[:, crop_stim == 3]
# %%
neurons = PC1_neg

all_arr = [np.mean(blank1[neurons, :], axis=1),
           np.mean(dot1[neurons, :], axis=1),
           np.mean(blank2[neurons, :], axis=1),
           np.mean(dot3[neurons, :], axis=1)]

sns.boxplot(data=all_arr)
plt.xticks([0, 1, 2, 3], ['blank 1', '1 dot', 'blank 1', '3 dots'])
plt.ylabel('dF/F')
plt.title(f'{fish_name}, sld.w {window}, prct. {percentile}')

# plt.savefig(f'{input_folder}/{fish_name}_boxplot_dff.png')
plt.show()

# %%
# %%
pc1 = 6
pc2 = 5

stim1 = X_pca[crop_stim == 0, :]
stim2 = X_pca[crop_stim == 1, :]
stim12 = X_pca[crop_stim < 2, :]

plt.plot(stim1[:, pc1], stim1[:, pc2], '.b')
plt.plot(stim2[:, pc1], stim2[:, pc2], '.r')

# plt.plot(stim12[:, pc1], stim12[:, pc2], 'k',linewidth=0.2)

plt.xlabel(f'PC {pc1}')
plt.ylabel(f'PC {pc2}')
plt.legend(['blank 1', 'dot 1'])
plt.title('PCA on Brain states')
plt.show()
# %%
sns.distplot(pca.components_[pc2, :])
plt.show()
# %%

