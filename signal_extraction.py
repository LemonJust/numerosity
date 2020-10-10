# %%
from IPython import get_ipython;

get_ipython().magic('reset -sf')

# %% setup
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


#  data loading

def read_tif(filename: str) -> np.ndarray:
    """Custom tif read function.

    Parameters
    ----------
    filename : string
        The path from which to read the image.

    Returns
    -------
    data : np.ndarray
        The image data in ZYX or TYX order
    """
    ext = os.path.splitext(filename)[-1]
    if ext in [".tif", ".tiff", ".lsm"]:
        import tifffile
        return tifffile.imread(filename)
    else:
        ValueError('Should be ".tif", ".tiff" or ".lsm"')


# %% load image
fish_name = 'fish1'
img_folder = 'D:/Code/repos/numerosity/Anna_20200717/img'
img_day = '190916'
image_file = f'{img_folder}/MAX_{img_day}_huch2bgcamp6s_52z_2p_1v3_{fish_name}.tif'
img = read_tif(image_file)

# load csv
input_folder = 'D:/Code/repos/numerosity/Anna_20200717/all_neurons'
csv_file = f'{input_folder}/{fish_name}/Results.csv'
points = pd.read_csv(csv_file, usecols=['X', 'Y'])
# %% loop over all the selected points
diam = 2
signal = np.zeros((len(points), img.shape[0]))

for index, row in points.iterrows():
    print(f"cell # {index}, XY: {row['X']}, {row['Y']}")
    # crop out image
    x, y = np.round([row['X'], row['Y']]).astype(int)
    cell = img[:, (y - diam):(y + diam), (x - diam):(x + diam)]
    signal[index, :] = np.mean(cell, axis=(1, 2))

print(f'N points : {len(points)}')
# %% plot all the traces

plt.plot(signal.T)
plt.title(f'Raw Signal Traces of Selected Cells, {fish_name}')
plt.xlabel('Time, frames')
plt.ylabel('I , a.u.')

plt.savefig(f'{input_folder}/{fish_name}_all_traces.png')
plt.show()

# %% save extracted signals
save_file = f'{input_folder}/{fish_name}/{fish_name}_signals_diam2_NxT.csv'
np.savetxt(save_file, signal, delimiter=',')

# %% how to load :
# b = np.loadtxt(save_file, delimiter=',', dtype=float)
# np.any(signal==b)
