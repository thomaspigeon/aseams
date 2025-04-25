#!python3

import numpy as np
from aseams.importance_sampling import infinite_integrale,find_x_newton, rayleigh
from scipy.integrate import quad
import scipy.integrate
import matplotlib.pyplot as plt

sigma = 0.05

print(infinite_integrale(lambda x :0.5+0.5*np.tanh( (x - 0.5)), sigma))

print(quad(lambda x : (0.5+0.5*np.tanh( (x - 0.5)))*rayleigh(x, sigma), 0,np.infty) )


bias_param={}
bias_param["committor_approx"] = lambda x: 0.5+0.5*np.tanh(x-0.5)

bias_param["norm"] =infinite_integrale(bias_param["committor_approx"], sigma)

print(bias_param["norm"] )
def cdf_val(vmax):
    return quad(lambda v : rayleigh(v, sigma) * bias_param["committor_approx"](v)/bias_param["norm"], 0,vmax)[0]

def pdf_val(v):
    return rayleigh(v, sigma) * bias_param["committor_approx"](v)/bias_param["norm"]

vmax=0.5
print(pdf_val(vmax))
vmax=find_x_newton(cdf_val, pdf_val, 1-0.1/1000, vmax)

print(vmax)

print(cdf_val(vmax))

bias_param["cdf_vels"] = np.linspace(0.0, vmax, 1000)
res_ivp = scipy.integrate.solve_ivp(lambda v, y: np.asarray([rayleigh(v, sigma) * bias_param["committor_approx"](v)/bias_param["norm"]]), [0, vmax], y0=[0.0], t_eval=bias_param["cdf_vels"])
bias_param["cdf"] =res_ivp.y.squeeze()

plt.plot(bias_param["cdf_vels"],pdf_val(bias_param["cdf_vels"]))

plt.plot(bias_param["cdf_vels"],bias_param["cdf"])
plt.grid()
plt.show()