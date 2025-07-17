import numpy as np
import os
from scipy.interpolate import interp1d

# Load real data
data_dir = 'tess_light_curves'
fluxes = []
labels = []
max_length = 0

for file in os.listdir(data_dir):
    if file.startswith('flux_'):
        tic_sector = file.replace('flux_', '').replace('.npy', '')
        flux = np.load(os.path.join(data_dir, file))
        label = np.load(os.path.join(data_dir, f'label_{tic_sector}.npy'))
        # Handle nans
        mask = np.isnan(flux)
        if np.any(mask):
            x = np.arange(len(flux))
            interp_func = interp1d(x[~mask], flux[~mask], bounds_error=False, fill_value='extrapolate')
            flux[mask] = interp_func(x[mask])
        # Normalize
        flux = (flux - np.nanmean(flux)) / np.nanstd(flux) if np.nanstd(flux) > 0 else flux - np.nanmean(flux)
        fluxes.append(flux)
        labels.append(label[0])
        max_length = max(max_length, len(flux))

# Pad or crop to max_length
padded_fluxes = np.zeros((len(fluxes), max_length))
for i, flux in enumerate(fluxes):
    if len(flux) > max_length:
        padded_fluxes[i] = flux[:max_length]  # Crop
    else:
        padded_fluxes[i, :len(flux)] = flux  # Pad

fluxes = padded_fluxes

# Data augmentation
augmented_fluxes, augmented_labels = [], []
for flux, label in zip(fluxes, labels):
    # Noise
    augmented_fluxes.append(flux + np.random.normal(0, 0.01, max_length))
    augmented_labels.append(label)
    # Shift
    shift = np.roll(flux, np.random.randint(100))
    augmented_fluxes.append(shift)
    augmented_labels.append(label)
# Add synthetic negatives
for _ in range(6):  # Add 6 to balance positives
    flux_syn = np.ones(max_length) + np.random.normal(0, 0.001, max_length)
    augmented_fluxes.append(flux_syn)
    augmented_labels.append(0)

fluxes = np.concatenate([fluxes, augmented_fluxes])
labels = np.concatenate([labels, augmented_labels])

# Resize to 2000 for CNN
max_len = 2000
fluxes_padded = np.zeros((len(fluxes), max_len))
for i, f in enumerate(fluxes):
    if len(f) > max_len:
        fluxes_padded[i] = f[:max_len]
    else:
        fluxes_padded[i, :len(f)] = f

np.save('processed_fluxes.npy', fluxes_padded)
np.save('labels.npy', labels)
print(f"Processed {len(fluxes_padded)} light curves (6 real + augmented/synthetic)!")