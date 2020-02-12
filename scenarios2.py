import numpy as np
import matplotlib.pyplot as plt
from compartment_model import dSIRdt_vec
from scipy.stats import poisson
from peak_ratio import month_lookup
from matplotlib.cm import plasma
from matplotlib.colors import to_hex


rec = 36   # 10 day serial interval
nb_pts = 4
R0s = np.linspace(1.3,2.5,nb_pts)
migrations = np.logspace(-3, -1, nb_pts)
N0,N1 = 6e7,1e8

fig, axes = plt.subplots(nb_pts,nb_pts, sharex=True, sharey=True, figsize=(14,10))

for ii,migration in enumerate(migrations):
    for jj,R0 in enumerate(R0s):

        # add Hubei population with parameters specified above
        #          population size, beta, rec, eps, theta, NH, containment, migration
        params = [[N0, R0*rec, rec, 0.4, 0.0, 1,     0.5, migration],
                  [N1, R0*rec, rec, 0.5, 10.5/12, 1, 0.5, migration],
                  [N1, R0*rec, rec, 0.5, 0.5/12, 1,  0.5, migration],
                  [N1, R0*rec, rec, 0.5, 2.5/12, 1,  0.5, migration]]
        # initially fully susceptible with one case in Hubei, no cases in NH
        populations = [[1, 1/N0], [1,0], [1,0], [1,0]]
        #total number of populations
        n_pops = len(params)


        params = np.array(params)
        populations = [np.array(populations)]

        # start simulation
        t = [2019.9]
        dt = 0.001
        tmax = 2021.5
        while t[-1]<tmax:
            dS, dI = dSIRdt_vec(populations[-1][:,0], populations[-1][:,1], t[-1], params)
            populations.append(populations[-1] + dt*np.array([dS,dI]).T)

            I_tot = (params[:,0]*populations[-1][:,1]).sum()
            populations[-1][:,1] += poisson.rvs(I_tot/n_pops*dt*params[:,7])/params[:,0]
            populations[-1][populations[-1][:,1]<1/params[:,0],1] = 0
            t.append(t[-1]+dt)

        populations = np.array(populations)


        colors = ['C0', to_hex(plasma(0.1)), to_hex(plasma(0.5)), to_hex(plasma(0.9))]

        fs=16

        axes[ii,jj].plot(t, populations[:,0,1]*params[0, 0], lw=3, label='Hubei', ls='--', c=colors[0])

        for pi in range(1,len(params)):
            axes[ii,jj].plot(t, populations[:,pi,1]*params[pi, 0], c=colors[pi],
                    lw=3, label=r"NE $\theta=$" + f'{month_lookup[int(params[pi,4]*12-0.5)]}')

        if (ii==0 and jj==nb_pts-1):
            axes[ii,jj].legend(fontsize=fs*0.8, loc="best")
        if jj==0:
            axes[ii,jj].set_ylabel('m=%.4f'%migration, fontsize=fs)
        if ii==nb_pts-1:
            axes[ii,jj].set_xticks(np.array([2020, 2020.25, 2020.5, 2020.75, 2021, 2021.25, 2021.5]))
            axes[ii,jj].set_xticklabels(['2020-01', '2020-04', '2020-07', '2020-10', '2021-01', '2021-04', '2021-07'])
            plt.setp(axes[ii,jj].xaxis.get_majorticklabels(), rotation=30)
            axes[ii,jj].set_xlabel(r"$\langle R_0\rangle=$"+"%.1f"%R0, fontsize=fs)
        axes[ii,jj].tick_params(labelsize=0.8*fs)
        axes[ii,jj].set_yscale('log')

plt.tight_layout()
plt.savefig('figures/scenario_subplot.pdf', format="pdf")
plt.show()

