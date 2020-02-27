import numpy as np
import matplotlib.pyplot as plt
from compartment_model import trajectory
from scipy.stats import poisson
from peak_ratio import month_lookup
from matplotlib.cm import plasma
from matplotlib.colors import to_hex


rec = 72   # 10 day serial interval
incubation = 5/365
nb_pts = 4
R0s = np.linspace(1.3,3.0,nb_pts)
migrations = np.logspace(-3, -1, nb_pts)
N0,N1 = 6e7,1e8
eps_hubei = 0.4
eps = 0.5

fig, axes = plt.subplots(nb_pts,nb_pts, sharex=True, sharey=True, figsize=(14,10))

for ii,migration in enumerate(migrations):
    for jj,R0 in enumerate(R0s):

        # add Hubei population with parameters specified above
        #          population size, beta, rec, eps, theta, NH, containment, migration
        params = [[N0, R0*rec, rec, eps_hubei, 0.0, 1,     0.5, migration, incubation],
                  [N1, R0*rec, rec, eps, 10.5/12, 1, 0.5, migration, incubation],
                  [N1, R0*rec, rec, eps, 0.5/12, 1,  0.5, migration, incubation],
                  [N1, R0*rec, rec, eps, 2.5/12, 1,  0.5, migration, incubation]]
        # initially fully susceptible with one case in Hubei, no cases in NH
        populations = [[1, 0, 1/N0], [1,0, 0], [1,0, 0], [1,0, 0]]
        #total number of populations
        n_pops = len(params)


        params = np.array(params)
        initial_population = np.array(populations)
        dt = 0.001
        t0 = 2019.8
        tmax = 2021.5
        # start simulation
        t, populations = trajectory(initial_population, t0, tmax, dt, params,
                                    resampling_interval=0, turnover=0)

        colors = ['C0', to_hex(plasma(0.1)), to_hex(plasma(0.5)), to_hex(plasma(0.9))]

        fs=16

        axes[ii,jj].plot(t, populations[:,0,1]*params[0, 0], lw=3, label='Hubei', ls='--', c=colors[0])

        for pi in range(1,len(params)):
            axes[ii,jj].plot(t, populations[:,pi,2]*params[pi, 0], c=colors[pi],
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
plt.savefig(f'figures/scenario_subplot_{eps}.pdf', format="pdf")
plt.show()

