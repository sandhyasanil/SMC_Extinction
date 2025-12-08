"""
Example usage of the ExtinctionCurve model.
"""
# Append parent directory to sys.path for imports
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import matplotlib.pyplot as plt
from model import ExtinctionCurve
from astropy import units as u


# Example wavelength grid (inverse micron)
x = np.linspace(1.0, 8.0, 500)*u.micron**-1

# Instantiate the model
curve = ExtinctionCurve(x, rv=3.1)


# Compute extinction curve
k = curve.evaluate()

# Plot
plt.figure()
plt.plot(x, k)
plt.xlabel(r"1/$\lambda$ ($\mu$m$^{-1}$)")
plt.ylabel(r"A($\lambda$)/A(V)")
plt.title("Example Extinction Curve")
plt.grid()
plt.show()

