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


def get_dff(signal, baseline):
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

# %% calculate and plot baseline and dff (blank>1>blank>3 5 sec each ==> 20 /cycle)
"""
Plot signal and baseline
"""
window = 21
percentile = 8
baseline = get_baseline(signal, window, percentile)
crop_sig = crop_to_window(signal.T, window).T

n_cell = 13
plt.subplot(2, 1, 1)
plt.plot(crop_sig[n_cell, :])
plt.plot(baseline[n_cell, :])
plt.xlabel('Time, frames')
plt.ylabel('Intensity, a.u.')
plt.title(f'{fish_name}, sld.w {window}, prct. {percentile}, cell {n_cell}, zoomed time')
plt.xlim(0, 250)
plt.legend(['raw signal', 'baseline'])

# plt.show()

"""
Get the dff and plot it :) 
"""
dff = get_dff(crop_sig, baseline)

plt.subplot(2, 1, 2)
plt.plot(dff[n_cell, :])
plt.xlabel('Time, frames')
plt.ylabel('dF/F')
plt.title(f'{fish_name}, sld.w {window}, prct. {percentile}, cell {n_cell}, zoomed time')
plt.xlim(0, 250)
plt.show()
plt.savefig(f'{input_folder}/{fish_name}_example_baseline_dff.png')

# %% Generate stimulus pattern
stim_pattern = '00000111112222233333'
stimulus = get_stim(stim_pattern, signal.shape[1])
crop_stim = crop_to_window(stimulus, window)
# %% plot box-plots for DFF
blank1 = dff[:, crop_stim == 0]
dot1 = dff[:, crop_stim == 1]
blank2 = dff[:, crop_stim == 2]
dot3 = dff[:, crop_stim == 3]

all_arr = [np.mean(blank1, axis=1),
           np.mean(dot1, axis=1),
           np.mean(blank2, axis=1),
           np.mean(dot3, axis=1)]

sns.boxplot(data=all_arr)
plt.xticks([0, 1, 2, 3], ['blank 1', '1 dot', 'blank 1', '3 dots'])
plt.ylabel('dF/F')
plt.title(f'{fish_name}, sld.w {window}, prct. {percentile}')

plt.savefig(f'{input_folder}/{fish_name}_boxplot_dff.png')
plt.show()
# %% plot box-plots for Raw Signal
blank1 = signal[:, stimulus == 0]
dot1 = signal[:, stimulus == 1]
blank2 = signal[:, stimulus == 2]
dot3 = signal[:, stimulus == 3]

all_arr = [np.mean(blank1, axis=1),
           np.mean(dot1, axis=1),
           np.mean(blank2, axis=1),
           np.mean(dot3, axis=1)]

sns.boxplot(data=all_arr)
plt.xticks([0, 1, 2, 3], ['blank 1', '1 dot', 'blank 2', '3 dots'])
plt.ylabel('Raw I, a.u.')
plt.title(f'{fish_name}, Raw Intensity ')
plt.savefig(f'{input_folder}/{fish_name}_boxplot_raw_signal.png')
plt.show()

# %% Output dff as table
blank1 = dff[:, crop_stim == 0]
dot1 = dff[:, crop_stim == 1]
blank2 = dff[:, crop_stim == 2]
dot3 = dff[:, crop_stim == 3]

n_neurons = signal.shape[0]
col_names = [f'cell_{x}' for x in range(1, n_neurons + 1)]
df1 = pd.DataFrame(blank1.T, columns=col_names)
df2 = pd.DataFrame(dot1.T, columns=col_names)
df3 = pd.DataFrame(blank2.T, columns=col_names)
df4 = pd.DataFrame(dot3.T, columns=col_names)

dfs = {'Blank1': df1, '1_dot': df2, 'Blank2': df3, '3_dots': df4}
writer = pd.ExcelWriter(f'{input_folder}/{fish_name}_dff_data.xlsx', engine='xlsxwriter')
for sheet_name in dfs.keys():
    dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
writer.save()
