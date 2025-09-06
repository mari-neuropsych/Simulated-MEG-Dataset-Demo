# Simulated MEG project generator (Jupyter cell)
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, butter, filtfilt

# ====== parameters ======
n_participants = 8
n_trials = 80
sfreq = 250                # Hz (downsampled)
epoch_tmin = 0.0
epoch_tmax = 0.5            # seconds
n_times = int((epoch_tmax - epoch_tmin) * sfreq)  # 125
n_channels = 10
channel_names = [f"MEG{i}" for i in range(n_channels)]

# bandpass helper
def bandpower_epochs(epoch_data, sfreq, low, high):
    # epoch_data: shape (n_channels, n_times)
    f, Pxx = welch(epoch_data, fs=sfreq, nperseg=min(256, epoch_data.shape[1]), axis=1)
    idx = np.logical_and(f >= low, f <= high)
    # integrate power in band per channel
    band_power = Pxx[:, idx].mean(axis=1)
    return band_power  # shape (n_channels,)

# create container for rows
rows = []

rng = np.random.default_rng(42)  # reproducible

for subj in range(1, n_participants + 1):
    # per-subject baseline SNR variability
    subj_snr = 0.5 + 0.5 * rng.random()
    for trial in range(1, n_trials + 1):
        # alternate conditions A/B roughly half-half
        condition = 'A' if (trial % 2 == 1) else 'B'
        # create noise: channels x times
        noise = rng.normal(loc=0.0, scale=1.0, size=(n_channels, n_times)) * (1.0 / subj_snr)

        # add an evoked-like bump centered ~150 ms (index ~0.15*sfreq)
        t = np.arange(n_times) / sfreq  # seconds from 0 to 0.5
        center = 0.15  # sec
        width = 0.04   # sec
        gauss = np.exp(-0.5 * ((t - center) / width)**2)
        # condition effect: A has slightly larger amplitude than B
        amp = 1.2 if condition == 'A' else 0.8
        # channel-specific topography (random but fixed per subject)
        topo = rng.normal(loc=1.0, scale=0.3, size=(n_channels, 1))
        evoked = (topo * amp) @ gauss.reshape(1, n_times)  # shape (channels, times)

        epoch = noise + evoked

        # compute features:
        # 1) per-channel mean amplitude (over epoch)
        mean_per_chan = epoch.mean(axis=1)  # shape (n_channels,)
        global_mean = mean_per_chan.mean()

        # 2) per-channel peak amp and latency (abs peak)
        abs_epoch = np.abs(epoch)
        peak_idx = abs_epoch.argmax(axis=1)  # time index of peak per channel
        peak_amp_per_chan = epoch[np.arange(n_channels), peak_idx]
        peak_latency_per_chan = peak_idx / sfreq  # seconds
        # global peak amplitude (max across channels by abs)
        ch_of_global_peak = np.argmax(np.abs(peak_amp_per_chan))
        global_peak_amp = peak_amp_per_chan[ch_of_global_peak]
        global_peak_latency = peak_latency_per_chan[ch_of_global_peak]

        # 3) band power (theta 4-7 Hz, alpha 8-12 Hz) average across channels
        theta_power_per_chan = bandpower_epochs(epoch, sfreq, 4, 7)
        alpha_power_per_chan = bandpower_epochs(epoch, sfreq, 8, 12)
        theta_power_global = theta_power_per_chan.mean()
        alpha_power_global = alpha_power_per_chan.mean()

        # 4) behavioral: simulate RT (ms) and accuracy (0/1)
        # make RT somewhat inversely related to global_mean (higher neural response -> faster)
        rt_mean = 450 - 50 * (global_mean)  # base ~450ms
        rt = rng.normal(loc=rt_mean, scale=40)
        if rt < 150: rt = 150.0
        # accuracy depends weakly on alpha power
        acc_prob = 0.6 + 0.3 * (alpha_power_global / (alpha_power_global + 1e-6))
        accuracy = 1 if rng.random() < acc_prob else 0

        # assemble row: participant-level + trial-level + global features + optional per-channel means
        row = {
            "participant": f"sub-{subj:02d}",
            "trial": trial,
            "condition": condition,
            "global_mean_amp": float(global_mean),
            "global_peak_amp": float(global_peak_amp),
            "global_peak_latency_s": float(global_peak_latency),
            "theta_power_global": float(theta_power_global),
            "alpha_power_global": float(alpha_power_global),
            "RT_ms": float(rt),
            "Accuracy": int(accuracy)
        }
        # add per-channel mean/peak if you want (prefix MEGmean_ / MEGpeak_)
        for ch_idx, chname in enumerate(channel_names):
            row[f"{chname}_mean"] = float(mean_per_chan[ch_idx])
            row[f"{chname}_peak"] = float(peak_amp_per_chan[ch_idx])
            row[f"{chname}_peak_latency_s"] = float(peak_latency_per_chan[ch_idx])
        rows.append(row)

# build DataFrame and save CSV
df = pd.DataFrame(rows)
# order columns: participant, trial, condition, globals..., then channels
cols = ["participant","trial","condition",
        "global_mean_amp","global_peak_amp","global_peak_latency_s",
        "theta_power_global","alpha_power_global","RT_ms","Accuracy"]
# append per-channel columns in order
for ch in channel_names:
    cols += [f"{ch}_mean", f"{ch}_peak", f"{ch}_peak_latency_s"]
df = df[cols]

csv_name = "simulated_meg_project.csv"
df.to_csv(csv_name, index=False)
print(f"Saved simulated dataset -> {csv_name}")
print("Shape (rows = trials total):", df.shape)
print("Example rows:")
print(df.head())
