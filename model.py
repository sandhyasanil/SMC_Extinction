import numpy as np

class ExtinctionCurve:
    
    from astropy import units as u
    """
    Extinction curve class implementing a parametric extinction law.

    Parameters
    ----------
    x : array_like
        Wavenumber array (1 / microns).
    rv : float, optional
        Total-to-selective extinction ratio. Default is 2.74.
    """

    def __init__(self, x, rv=2.74):
        self.x = x
        self.rv = rv

    #--------------------------
    #Input unit validation
    #-------------------------
    def validate_input_units(self):
        """
        Verify if input has units specified. If yes then ensure it is inverse microns. Else throw warning.

        Raises
        ------
        ValueError
            If the input unit is not '1/micron'.
        """

        if isinstance(self.x, self.u.Quantity):
            if self.u.get_physical_type(self.x.unit) == 'length':
                # self.x = 1/self.x.to(self.u.micron)
                # print(self.x.unit)
                return(1/self.x.to(self.u.micron))
            elif self.u.get_physical_type(self.x.unit) == 'wavenumber':
                # self.x = self.x.to(1/self.u.micron)
                # print(self.x.unit)
                return(self.x.to(1/self.u.micron))
            else:
                raise ValueError("Input unit must be of type 'length' or 'wavenumber'.")
        else:
            raise ValueError("Input unit not specified.")


    # -------------------------
    # UV correction terms
    # -------------------------
    def fa(self, x):
        """UV correction term for A(x) at x >= 5.9."""
        return np.where(
            x >= 5.9,
            1.0004e-03 * (x - 5.9)**2 + 1e-04 * (x - 5.9)**3,
            0.0
        )

    def fb(self, x):
        """UV correction term for B(x) at x >= 5.9."""
        return np.where(
            x >= 5.9,
            2.8548e-01 * (x - 5.9)**2 + 1e-04 * (x - 5.9)**3,
            0.0
        )

    # -------------------------
    # A(x) and B(x) components
    # -------------------------
    def a(self, x):
        x = np.asarray(x)
        y = np.zeros_like(x)

        # Region 1: IR (x <= 1.1)
        mask1 = x <= 1.1
        y[mask1] = 0.574 * x[mask1]**1.61

        # Region 2: Optical/NIR (1.1 < x <= 3.2429)
        mask2 = (x > 1.1) & (x <= 3.2429)
        y2 = x[mask2] - 1.82
        y[mask2] = (
            1
            + 0.17699*y2 - 0.50447*y2**2 - 0.02427*y2**3
            + 0.72085*y2**4 + 0.01979*y2**5 - 0.77530*y2**6 + 0.32999*y2**7
        )

        # Region 3: UV (x > 3.2429)
        mask3 = x > 3.2429
        y[mask3] = (
            -1.10895637
            + 0.53314169 * x[mask3]
            + 0.017348 / ((x[mask3] - 4.6)**2 + 0.2444)
            + self.fa(x[mask3])
        )

        return y

    def b(self, x):
        x = np.asarray(x)
        y = np.zeros_like(x)

        # Region 1: IR
        mask1 = x <= 1.1
        y[mask1] = -0.527 * x[mask1]**1.61

        # Region 2: Optical/NIR
        mask2 = (x > 1.1) & (x <= 3.2429)
        y2 = x[mask2] - 1.82
        y[mask2] = (
            1.41338*y2 + 2.28305*y2**2 + 1.07233*y2**3
            - 5.38434*y2**4 - 0.62251*y2**5 + 5.30260*y2**6 - 2.09002*y2**7
        )

        # Region 3: UV
        mask3 = x > 3.2429
        y[mask3] = (
            1.01154931
            + 0.76809458 * x[mask3]
            + 0.047807 / ((x[mask3] - 4.6)**2 + 0.2444)
            + self.fb(x[mask3])
        )

        return y

    # -------------------------
    # Full extinction curve
    # -------------------------
    def evaluate(self):
        """
        Evaluate the extinction curve A(λ)/A(V).

        Parameters
        ----------
        x : array_like, optional
            Wavenumber array. If None, uses the stored x values.

        Returns
        -------
        ndarray
            Extinction curve A(λ)/A(V).
        """
        # if x is None:
        #     x = self.x
        x = self.validate_input_units()
        print(x)
        print(x.unit)
        x = np.asarray(x)
        return self.a(x) + self.b(x) / self.rv
