import numpy as np
import matplotlib.pyplot as plt
from compartment_model import trajectory
from scipy.stats import poisson
from peak_ratio import month_lookup

if __name__ == '__main__':

    rec = 72   # 10 day serial interval
    migration = 1e-2 # rate of moving per year
    N0,N1 = 6e7,1e8
    incubation_time = 5/365
    eps_hubei = 0.4
    eps = 0.5

    # add Hubei population with parameters specified above
    #          population size, beta, rec, eps, theta, NH, containment, migration
    params = [[N0, 2.2*rec, rec, eps_hubei, 0.0, 1,     0.5, migration, incubation_time],
              [N1, 2.2*rec, rec, eps, 10.5/12, 1, 0.5, migration, incubation_time],
              [N1, 2.2*rec, rec, eps, 0.5/12, 1,  0.5, migration, incubation_time],
              [N1, 2.2*rec, rec, eps, 2.5/12, 1,  0.5, migration, incubation_time]]
    # initially fully susceptible with one case in Hubei, no cases in NH
    populations = [[1, 0, 1/N0], [1,0,0], [1,0,0], [1,0,0]]
    #total number of populations
    n_pops = len(params)


    params = np.array(params)
    initial_population = np.array(populations)

    # start simulation
    dt = 0.001
    t0 = 2019.8
    tmax = 2021.5
    t, populations = trajectory(initial_population, t0, tmax, dt, params,
                                resampling_interval=0, turnover=0)

    from matplotlib.cm import plasma
    from matplotlib.colors import to_hex
    colors = ['C0', to_hex(plasma(0.1)), to_hex(plasma(0.5)), to_hex(plasma(0.9))]

    fs=16
    plt.figure()
    plt.plot(t, populations[:,0,2]*params[0, 0], lw=3, label='Hubei', ls='--', c=colors[0])

    for pi in range(1,len(params)):
        plt.plot(t, populations[:,pi,2]*params[pi, 0], c=colors[pi],
                lw=3, label=r"NE $\theta=$" + f'{month_lookup[int(params[pi,4]*12-0.5)]}')

    plt.legend(fontsize=fs*0.8, loc=8, ncol=2)
    plt.yscale('log')
    plt.ylabel('Cases', fontsize=fs)
    plt.xticks(np.array([2020, 2020.25, 2020.5, 2020.75, 2021, 2021.25]),
               ['2020-01', '2020-04', '2020-07', '2020-10', '2021-01', '2021-04'])
    plt.tick_params(labelsize=0.8*fs)
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig('figures/scenarios.pdf')

