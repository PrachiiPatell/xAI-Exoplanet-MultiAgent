import lightkurve as lk
import numpy as np
import os

# Create folder for data
os.makedirs('tess_light_curves', exist_ok=True)

# Positives (your existing)
positives = [
    ('261136679', 1, 1),
    ('231702397', 1, 1),
    ('55525572', 1, 1),
    ('350153977', 1, 1),
    ('234994474', 1, 1)
]

# Additional negatives (verified with data in Sector 1)
negatives = [
    ('425933644', 1, 0),  # Already downloaded
    ('176954932', 1, 0),
    ('300160320', 1, 0),
    ('149542119', 1, 0),
    ('220478335', 1, 0)
]

examples = positives + negatives

# Download loop
for tic, sector, label in examples:
    filename = f'tess_light_curves/time_{tic}_sector{sector}.npy'
    if os.path.exists(filename):
        print(f"Already downloaded: TIC {tic}, Sector {sector}, Label {label}")
        continue
    try:
        search = lk.search_lightcurve(f'TIC {tic}', mission='TESS', sector=sector)
        if len(search) > 0:
            lc = search[0].download()
            time_data = lc.time.jd
            flux_data = lc.flux.filled(np.nan)
            time = time_data.value if hasattr(time_data, 'value') else time_data
            flux = flux_data.value if hasattr(flux_data, 'value') else flux_data
            np.save(f'tess_light_curves/time_{tic}_sector{sector}.npy', time)
            np.save(f'tess_light_curves/flux_{tic}_sector{sector}.npy', flux)
            np.save(f'tess_light_curves/label_{tic}_sector{sector}.npy', np.array([label]))
            print(f"Downloaded: TIC {tic}, Sector {sector}, Label {label}")
        else:
            print(f"No data for TIC {tic}, Sector {sector}")
    except Exception as e:
        print(f"Error for TIC {tic}: {e}")

print("Download complete—rerun preprocessing.")