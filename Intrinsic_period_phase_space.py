import numpy as np
import matplotlib.pyplot as plt

def get_period(B,N,b):
    R0 = B/(N+b)
    delta = R0**2*b**2-4*b*(N*(R0-1)-b)
    period = np.zeros(delta.shape)
    period[delta<0] = (4.0*np.pi)/np.sqrt(-delta[delta<0])
    return period


# Parameters
R0 = 2.2
nu = 36
beta = nu*R0
nu_min, nu_max = 0.5*nu, 1.5*nu
beta_min, beta_max = 0.5*beta, 1.5*beta
b=0.2
nb_pts = 100
betas = np.linspace(beta_min, beta_max, nb_pts)
nus = np.linspace(nu_min, nu_max, nb_pts)
B, N = np.meshgrid(betas, nus)
levels = np.linspace(0,10,40+1)
fontsize = 16

fig = plt.figure()
CS = plt.contourf(B, N, get_period(B,N,b), levels=levels)
CS2 = plt.contour(CS, levels=[1,2,3,4], colors='r', origin="lower")
plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=0.7*fontsize)
c = plt.colorbar(CS)
c.add_lines(CS2)
c.ax.tick_params(labelsize=0.7*fontsize)
c.set_label(r"$T$", fontsize=fontsize)
plt.xlabel(r"$\beta$", fontsize=fontsize)
plt.ylabel(r"$\nu$", fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"figures/Period_phase_space_b_{b}.pdf", format='pdf')
#plt.show()