import numpy as np
import matplotlib.pyplot as plt
import math

def get_period(B,N,b):
    R0 = B/(N+b)
    delta = R0**2*b**2-4*b*(N*(R0-1)-b)
    period = np.zeros(delta.shape)
    period[delta<0] = (4.0*math.pi)/np.sqrt(-delta[delta<0])
    return period


# Parameters
beta_min, beta_max = 0.5*72, 1.5*72
nu_min, nu_max = 0.5*36, 1.5*36
b=0.1
nb_pts = 100
betas = np.linspace(beta_min, beta_max, nb_pts)
nus = np.linspace(nu_min, nu_max, nb_pts)
B, N = np.meshgrid(betas, nus)
levels = np.linspace(0,10,40+1)
fontsize = 18

fig = plt.figure(figsize=(14,10))
CS = plt.contourf(B, N, get_period(B,N,b), levels=levels)
CS2 = plt.contour(CS, levels=[1,2,3,4], colors='r', origin="lower")
plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=0.7*fontsize)
c = plt.colorbar(CS)
c.add_lines(CS2)
c.ax.tick_params(labelsize=0.7*fontsize)
c.set_label(r"$T$", fontsize=fontsize)
plt.xlabel(r"$\beta$", fontsize=fontsize)
plt.ylabel(r"$\nu$", fontsize=fontsize)
plt.savefig("figures/Period_phase_space.pdf", format='pdf')
#plt.show()