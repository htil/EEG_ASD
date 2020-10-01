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


from EEGRecording import EEGRecording
from utils import visualize
# from Pipeline import Pipeline

### CONFIGURATION ###
dir = Path('.')  # directory for eeg_asd data
include256Hz = True  # Shouldn't matter b/c we are doing frequency-domain analysis
# If set to True, the model will only run on those chanels that are shared by the caps. should be 22 also.
include24Cap = False
# Options: "NONE", "SKLEARN_MIN_MAX_SCALER" TODO: apply to new pipeline
normalizeStrat = "NONE"
chanelsToInclude = np.ones(34)  # TODO: apply to new pipeline


class Pipeline:

    def __init__(self):
        self.pipeline = []

    def load_data(self):
        # IMPORT THE DATA
        # Clean out the previous files

        for file in dir.glob('./raw_data/*/*'):
            # Create a pandas DataFrame for the file
            df = pd.read_csv(file, sep='\t', index_col=False)
            df = df.iloc[:, :-1]  # Remove last column (just a csv read error)
            name = os.path.basename(file).split('_')
            freq = name[2].split('.')[0].replace('Hz', '').replace('HZ', '')
            new_filename = f'{name[0]}_{freq}.csv'
            self.pipeline.append(EEGRecording(
                new_filename, freq, df.columns.size, new_filename[0], df))

        return self

    # TODO: use both 256 Hz and 512 Hz
    # TODO: use 24-chan and 32-chan

    def find_common_chanels(self):
        # 1. Find the common chanels in the data, remove uncommon ones -OR- remove 24-cap headests. See which method is preferable.
        # For now I'm just throwing out the 24 chan data, but that should def be fixed asap (TODO )
        if include24Cap:
            pass
        else:
            [self.pipeline.remove(p) for p in self.pipeline if p.chan == 22]
        return self

    # 2. Split into epochs

    def split_into_epochs(self):
        for p in self.pipeline:
            data = p.data.to_numpy()
            size = data.shape[0]
            # Todo: make the new epochs to 5 seconds.
            # 300 seconds/ 60. split into 60
            #
            quarter = math.floor(size / 4)
            newData = [pd.DataFrame(data[:quarter], columns=p.data.columns),
                       pd.DataFrame(data[quarter+1:2*quarter],
                                    columns=p.data.columns),
                       pd.DataFrame(
                           data[2*quarter+1:3*quarter, :], columns=p.data.columns),
                       pd.DataFrame(data[3*quarter+1:, :], columns=p.data.columns)]
            p.data = newData
        return self

    # 2.5 - look at each epcoh. Throw out hte ones that are just noise...

    # find a way to plot each one

    # eeg lab kind of helps with this if i want to look into tthat

    # distant future - automatically reject bad epochs....

    # 3. Welch & Powerband on each individual, then each epoch, then each EEG channel, then each band = N x 4 x 34 x 4.

    def do_welch_powerband(self):
        new_pipeline = []
        for p in self.pipeline:
            epoch_vals = []
            for epoch in p.data:
                channel_power_stats = []
                for col in epoch.columns:
                    # print(f"we are running on file {p.name}, on an epoch in there, on column name {col}")
                    data = epoch[col]
                    f = data.to_numpy()
                    sf = float(p.freq)

                    time = np.arange(f.size) / sf
                    win = 4 * sf
                    # TODO: learn the math behind this
                    freqs, psd = signal.welch(data, sf, nperseg=win)

                    # Let's find powers using band limits from Dr. Malaia's data description...

                    # Delta Power
                    low, high = 0.1, 4
                    idx_delta = np.logical_and(freqs >= low, freqs <= high)
                    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
                    delta_power = simps(psd[idx_delta], dx=freq_res)

                    # Theta Power
                    low, high = 4, 7
                    idx_theta = np.logical_and(freqs >= low, freqs <= high)
                    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
                    theta_power = simps(psd[idx_theta], dx=freq_res)

                    # alpha Power
                    low, high = 8, 12
                    idx_alpha = np.logical_and(freqs >= low, freqs <= high)
                    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
                    alpha_power = simps(psd[idx_alpha], dx=freq_res)

                    # Beta power
                    low, high = 15, 30
                    idx_beta = np.logical_and(freqs >= low, freqs <= high)
                    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
                    beta_power = simps(psd[idx_beta], dx=freq_res)

                    # Add this to the channel_power_stats, which will endup being 34x4
                    channel_power_stats.append(
                        [delta_power, theta_power, alpha_power, beta_power])
                # Now we can replace this epoch's data value with the 34x4 2d array we generated, and add it to the 4x34x4 3d array for this particular individual's recording
                epoch_vals.append(channel_power_stats)
            # Add our new EEG recording to the new pipeline, getting a Nx4x34x4 array
            new_pipeline.append(EEGRecording(
                p.name, p.freq, p.chan, p.clas, epoch_vals))
        self.pipeline = new_pipeline

    def save_state(self, filename):
        np.save(dir/f'data/{filename}', self.pipeline)  # Save it!

    def clear_all_saves(self):
        [os.remove(f) for f in dir.glob('./data/*')]

    # recording_number = 0
    # epoch_number = 0            # 0 = first quarter, etc.
    # channel_number = 0         # Use the pandas dataframe above for reference
    # band_number = 0             # 0 = Delta, 1 = Theta, 2 = Alpha, 3 = Beta
    # x = access_point(recording_number, epoch_number, channel_number, band_number)
    # print(f"recoridng #{recording_number}, epoch #{epoch_number}, channel #{chanel_number}, band #{band_number} "f" has a value of {}")
    def access_point(self, recording_number=0, epoch_number=0, chanel_number=0, band_number=0):
        return self.pipeline[recording_number].data[epoch_number][chanel_number][band_number]

    # Spectrogram...
