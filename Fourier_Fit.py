import numpy as np

class FourierSeries():    
    def __init__(self):
        """
        FourierSeries is a class that generates and fits Fourier series to time series data.
        """
        None
        

    def generate(self, x, params, e_params=None):
        """
        Generate a Fourier series of a given order.

        Args:
            x (array-like): Phase folded time series
            params (float): Parameters used to compute the Fourier series coefficients

        Returns:
            y (array-like): Sum of the Fourier series evaluated at the given time series
        """
        
        
        if isinstance(params, list):
            params = np.array(params)
        if isinstance(e_params, list):
            e_params = np.array(e_params)
            
        # check if params are provided if none raise error with suggestions of parameters
        if len(params) == 0:
            raise ValueError('No parameters provided. Please provide parameters. \n' 
                                'For example: params = [m0, A1, phi1, A2, phi2, ...] \n'
                                'where m0 is the mean, A1, A2, ... are the amplitudes and phi1, phi2, ... are the phases.')
        
        # check if number of parameters is odd
        if len(params) % 2 == 0:
            raise ValueError('Number of parameters must be odd.')
        
        y = self._generate(x, *params)
        
        if e_params is not None:
            # print('Error on parameters provided.')
            # print(params+e_params)
            # print(params-e_params)
            if len(e_params) != len(params):
                raise ValueError('Length of e_params must be equal to length of params.')
            else:
                y1 = self._generate(x, *(params + e_params))
                y2 = self._generate(x, *(params - e_params))                
                ey1, ey2 = abs(y1 - y), abs(y2 - y)
                ey = (ey1 + ey2)
        else:
            ey = None
        
        return y, ey
    
    def _generate(self, x, *params):
        """
        Generate a Fourier series of a given order.

        Args:
            x (array-like): Phase folded time series
            params (float): Parameters used to compute the Fourier series coefficients

        Returns:
            y (array-like): Sum of the Fourier series evaluated at the given time series
        """
        if isinstance(params, list):
            params = np.array(params)
            
        # check if params are provided if none raise error with suggestions of parameters
        if len(params) == 0:
            raise ValueError('No parameters provided. Please provide parameters. \n' 
                                'For example: params = [m0, A1, phi1, A2, phi2, ...] \n'
                                'where m0 is the mean, A1, A2, ... are the amplitudes and phi1, phi2, ... are the phases.')
        
        # check if number of parameters is odd
        if len(params) % 2 == 0:
            raise ValueError('Number of parameters must be odd.')

        N = len(params)//2
        m0, Ak, phik = params[0], np.array(
            params[1:N+1]), np.array(params[N+1:])
        y = m0

        for k in range(1, N+1):
            y += Ak[k-1]*np.sin(2*np.pi*k*x + phik[k-1])
        
        return y

    def bart_condition(self, n):
        """
        Calculate Bart's condition, which is a condition that must be satisfied for a Fourier series to be unique.

        Args:
            n (int): Order of the Fourier series

        Returns:
            cond (int): Bart's condition
        """
        cond = 2*n + 1
        return cond

    def fit(self, x, y, n, yerr=None, p0=None, return_ratio=False):
        """
        Fit a Fourier series to a time series.

        Args:
            x (array-like): Phase folded time series [0, 1]
            y (array-like): Corresponding flux time series
            n (int): Order of the Fourier series to be fitted
            yerr (array-like, optional): Uncertainties on the flux time series
            p0 (array-like, optional): Initial guess for the parameters
            return_ratio (bool, optional): Whether to return the ratio of the errors on the parameters

        Returns:
            params (array-like): Parameters of the fitted Fourier series
            e_params (array-like): Errors on the parameters of the Fourier series
            ratio (array-like, optional): Ratio of the errors on the parameters of the Fourier series
        """
        if p0 is None:
            p0 = np.array([np.mean(y)] + list(np.random.normal(0, 1, n)
                                              ) + list(np.random.normal(0, 2, n)*np.pi))

        if len(p0) != 2*n + 1:
            raise ValueError('p0 must have length 2*n + 1')
        
        from scipy.optimize import curve_fit

        params, cov = curve_fit(self._generate, x, y, sigma=yerr, p0=p0, method='lm')
        e_params = np.sqrt(np.diag(cov))
        
        m0, A, phi = params[0], params[1:n+1], params[n+1:]

        for k in range(len(A)):
            if A[k] < 0:
                A[k] = -A[k]
                phi[k] = phi[k] + np.pi
            if phi[k] < 0:
                while True:
                    phi[k] = phi[k]+2.0*np.pi
                    if (phi[k] > 0):
                        break
            if phi[k] > 2*np.pi:
                while True:
                    phi[k] = phi[k]-2.0*np.pi
                    if (phi[k] < 2*np.pi):
                        break

        params = np.array([m0] + list(A) + list(phi))

        if return_ratio:
            fp, e_fp = self.fparams_ratio(params, e_params)
            return params, e_params, fp, e_fp
        else:
            return params, e_params

    def fparams_ratio(self, params, e_params):
        """
        Calculate the ratio of the errors on the parameters of a Fourier series.

        Args:
            params (array-like): Parameters of the Fourier series
            e_params (array-like): Errors on the parameters of the Fourier series

        Returns:
            fp (array-like): Ratio of the errors on the parameters of the Fourier series
            e_fp (array-like): Errors on the ratio of the errors on the parameters of the Fourier series
        """
        # convert params and eparams to numpy array after checking if they are lists
        if isinstance(params, list):
            params = np.array(params)
        if isinstance(e_params, list):
            e_params = np.array(e_params)
        
        
        N = len(params)//2
        Ak = params[1:N+1]
        e_Ak = e_params[1:N+1]
        phik = params[N+1:]
        e_phik = e_params[N+1:]

        A_rat = Ak/Ak[0]
        phi_rat = phik - [(k+1)*phik[0] for k in range(0, N)]

        e_A_rat = np.sqrt(
            (e_Ak**2)/(Ak[0]**2) + (Ak**2)*(e_Ak[0]**2)/(Ak[0]**4))
        e_phi_rat = np.sqrt(
            e_phik**2 + [((k+1)**2)*(e_phik[0]**2) for k in range(0, N)])

        for k in range(len(A_rat)):
            if A_rat[k] < 0:
                A_rat[k] = -A_rat[k]
                phi_rat[k] = phi_rat[k] + np.pi
            if phi_rat[k] < 0:
                while True:
                    phi_rat[k] = phi_rat[k]+2.0*np.pi
                    if (phi_rat[k] > 0):
                        break
            if phi_rat[k] > 2*np.pi:
                while True:
                    phi_rat[k] = phi_rat[k]-2.0*np.pi
                    if (phi_rat[k] < 2*np.pi):
                        break

        fp = np.array([A_rat, phi_rat])
        e_fp = np.array([e_A_rat, e_phi_rat])

        return fp, e_fp

    def iterative_fit(self, x, y, n, trials=10, A_guess=None, phi_guess=None):
        """
        Iteratively fit a Fourier series to a time series.

        Args:
            x (array-like): Phase folded time series
            y (array-like): Corresponding flux time series
            n (int): Order of the Fourier series to be fitted
            trials (int, optional): Number of trials to perform
            A_guess (array-like, optional): Initial guess for the amplitude parameters
            phi_guess (array-like, optional): Initial guess for the phase parameters

        Returns:
            params (array-like): Parameters of the best-fit Fourier series
            e_params (array-like): Errors on the parameters of the best-fit Fourier series
            residual_mean (array-like): Mean of the absolute residuals for each trial
            r2 (array-like): R^2 value for each trial
            chi2_red (array-like): Reduced chi^2 value for each trial
        """
        residual_mean = np.empty(shape=(trials))
        r2 = np.empty(shape=(trials))
        chi2_red = np.empty(shape=(trials))
        params_array = np.empty(shape=(2*n+1, trials))
        e_params_array = np.empty(shape=(2*n+1, trials))

        for i in range(trials):
            if A_guess is None:
                A_guess = np.random.normal(
                    0, 1, 1)*np.logspace(1, -1, n)/(10*(i+1))
            if phi_guess is None:
                phi_guess = np.random.normal(0, 2, n)*np.pi

            p0 = np.array([np.mean(y)] + list(A_guess) + list(phi_guess))

            _params, _e_params = self.fit(x, y, n, p0=p0)

            residual_mean[i] = np.mean(np.abs(y - self.generate(x, *_params)))
            r2[i] = 1 - np.sum((y - self.generate(x, *_params))
                               ** 2)/np.sum((y - np.mean(y))**2)
            params_array[:, i] = _params
            e_params_array[:, i] = _e_params

        best_fit = np.argmax(r2)
        params = params_array[:, best_fit]
        e_params = e_params_array[:, best_fit]

        output = {
            'params': params,
            'e_params': e_params,
            'residual_mean': residual_mean,
            'r2': r2,
            'chi2_red': chi2_red
        }

        return output
    


# generate a test for FourierSeries.generate
import numpy as np
import matplotlib.pyplot as plt

FS = FourierSeries()

x = np.random.uniform(0, 1, 8)
x = np.sort(x)

m0, Ak, phik = 1, np.array([1, 0.5, 0.25]), np.array([0, np.pi/2, np.pi/4])
em0, eAk, ephik = 0.1, np.array([0.1, 0.5, 0.0025]), np.array([0.1, 0.05, 0.025])

params = np.array([m0] + list(Ak) + list(phik))
e_params = np.array([em0] + list(eAk) + list(ephik))

y, ey = FS.generate(x, params=params, e_params=e_params)

fig = plt.figure(figsize=(4,4))
plt.errorbar(x, y, yerr=ey, fmt='o-', capsize=2)

params1, e_params1 = FS.fit(x, y, 3, yerr=ey)
_x = np.linspace(0, 1, 20)
y1, ey1 = FS.generate(_x, params=params1, e_params=e_params1)

plt.errorbar(_x, y1, yerr=ey1, fmt='o-', capsize=2)
plt.errorbar(_x, FS.generate(_x, params=params)[0], fmt='o-', capsize=2)
plt.show()