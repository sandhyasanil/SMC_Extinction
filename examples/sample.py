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
curve_1 = ExtinctionCurve(x, rv=3.4)
curve_2 = ExtinctionCurve(x, rv=2.74)
curve_3 = ExtinctionCurve(x, rv=2.0)


# Compute extinction curve
k_1 = curve_1.evaluate()
k_2 = curve_2.evaluate()
k_3 = curve_3.evaluate()
# Plot
plt.figure()
plt.plot(x, k_1, label=r'$R_V$=3.4')
plt.plot(x, k_2, label=r'$R_V$=2.74')
plt.plot(x, k_3, label=r'$R_V$=2.0')
plt.xlabel(r"1/$\lambda$ ($\mu$m$^{-1}$)")
plt.ylabel(r"A($\lambda$)/A(V)")
plt.title("Example Extinction Curve")
plt.grid()
plt.legend()
plt.show()

