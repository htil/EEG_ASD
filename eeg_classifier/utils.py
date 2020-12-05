import pandas as pd
from pathlib import Path
import glob
from dataclasses import dataclass
from sklearn import preprocessing
import os
import math
import mne
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.integrate import simps

### UTILITY FUNCTIONS & DATA OBJECTS ###


def visualize(file, sfreq):
    # Create a pandas DataFrame for the file
    df = pd.read_csv(file, sep='\t', index_col=False)
    df = df.iloc[:, :-1]  # Remove last column (just a csv read error)
    data = df.to_numpy()  # Get a numpy copy of the data
    chNames = df.columns.to_list()  # Pull out column names for labels
    # Transpose data (mne uses rows to represent each channel, but .csv uses columns)
    dataT = data.T
    # Create the info structure needed by MNE
    info = mne.create_info(chNames, sfreq, 'eeg')
    raw = mne.io.RawArray(dataT, info)  # Create the raw MNE object
    # raw.plot() # Plot raw data PSD and first chanel for 3 and 10 secs
    mne.viz.plot_raw_psd(raw, fmin=0, fmax=30)

    # mne.viz.plot_raw(raw, n_channels=1,scalings='auto', clipping=None, start=0, duration=3.0)
    # mne.viz.plot_raw(raw, n_channels=1,scalings='auto', clipping=None, start=0, duration=10.0)
