import tensorflow as tf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Agent 1: Detection
def detect_planet(flux):
    model = tf.keras.models.load_model('planet_detector.keras')
    flux = flux.reshape(1, 2000, 1)
    pred = model.predict(flux)
    return pred[0][0] > 0.5

# Agent 2: Validation (enhanced with data-driven estimates)
def validate_orbit(candidate, flux):
    if not candidate:
        return "No candidate to validate"
    transit_depth = max(0, -(np.nanmin(flux) - np.nanmean(flux)))
    dips, _ = find_peaks(-flux, prominence=0.1 * transit_depth, distance=50)
    period_est = np.mean(np.diff(dips)) * 0.02 if len(dips) >= 2 else "Unknown"
    r_star = sp.symbols('r_star')
    r_planet = sp.sqrt(transit_depth) * r_star
    return f"Valid orbit: Planet radius ~{r_planet.subs(r_star, 1):.4f} stellar radii, depth {transit_depth:.4f}, period ~{period_est}"

# Agent 3: Reporting
def generate_report(validation, habitability, flux):
    plt.plot(flux.flatten())
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Flux')
    plt.title('Exoplanet Light Curve Report')
    plt.savefig('report.png')
    plt.close()
    return f"Report: {validation} | {habitability} | See report.png for plot"

# Agent 4: Habitability Assessor
def assess_habitability(validation, flux):
    try:
        radius = float(validation.split('radius ~')[1].split(' ')[0])
        period_str = validation.split('period ~')[1].split(' ')[0]
        period = float(period_str) if period_str != "Unknown" else 0
    except:
        radius, period = 0, 0
    if 0.5 < radius < 2 and 100 < period < 400:
        return "Potentially habitable: Earth-like size in habitable zone."
    return f"Not habitable: Extreme size/orbit (radius {radius:.4f}, period {period:.2f} days)."

# Multi-Agent Flow
fluxes = np.load('processed_fluxes.npy')
sample_flux = fluxes[0].flatten()  # Test a positive
candidate = detect_planet(sample_flux)
validation = validate_orbit(candidate, sample_flux)
habitability = assess_habitability(validation, sample_flux)
report = generate_report(validation, habitability, sample_flux)
print(report)