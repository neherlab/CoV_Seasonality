"""
script to run a simple seaonal SIR model with coupling to a global reservoir
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dSIRdt(tSEI, beta, eps, theta, rec, migration, frac_exposed, turnover, incubation):
    '''
    time derivative of a simple SIR model with an influx of infected
    (e.g. holiday returnees) and population turnover. Seasonality is
    modelled as harmonically oscillating infectivity with amplitude
    eps and peak at theta
    '''
    infection = beta*(1+eps*np.cos(2*np.pi*(tSEI[0]-theta)))*tSEI[1]*tSEI[3]

    dS = - infection + (1-tSEI[1])*turnover - migration*frac_exposed*tSEI[1]
    dE =  infection - (turnover+1/incubation)*tSEI[2] + migration*frac_exposed*tSEI[1]
    dI =  tSEI[2]/incubation - (turnover+rec)*tSEI[3]
    return np.array([1, dS, dE, dI])


def run_SIR(X, tmax, dt):
    '''
    generate a trajectory for parameters X of length tmax with integration
    time step dt
    '''
    beta, eps, theta, rec, migration, frac_exposed, turnover, incubation = X

    # initialize trajectory at steady state.
    tSEI = [np.array([0, rec/beta, 0, turnover*(R0-1)/(R0*rec)])]

    while tSEI[-1][0]<tmax:
        dSEIR = dSIRdt(tSEI[-1], beta, eps, theta, rec, migration, frac_exposed, turnover, incubation)
        tSEI.append(tSEI[-1] + dSEIR*dt)

    return np.array(tSEI)


def cost(traj, prevalence, plot=False, dt=0.001):
    avg = np.mean(np.array([[traj[-int((y+(11.5-m)/12)/dt),3] for y in range(10)]
            for m in range(12)]), axis=1)

    C = np.sum((avg/avg.mean() - prevalence/prevalence.mean())**2)
    if plot:
        plt.plot(avg/avg.mean())
        plt.plot(prevalence/prevalence.mean())
    print(C)
    return C


if __name__ == '__main__':
    # read average seasonal test-positive rate.
    df = pd.read_csv('data/frac_positive_by_month.tsv', sep='\t')
    from itertools import product
    # all time units are years. A rate of 1/day is hence 365/year
    # 1/week is 52/year

    R0 = 3
    rec = 72
    incubation = 5/365
    eps = 0.5
    theta = -0.15
    migration = 1e-3
    frac_exposed = 0.02
    turnover = 0.3 # rate at which people turn susceptible (0.1 corresponds to once every 10 years)
    dt=0.001
    tmax = 30

    X0 = (R0*rec, eps, theta, rec, migration/frac_exposed, frac_exposed, turnover, incubation)
    traj = run_SIR(X0, tmax, dt=dt)
    prevalence = df["all CoVs"]
    cost(traj, prevalence, dt=dt)
    plt.plot(traj[-3000:,0], traj[-3000:,2])
    plt.ylim([0,0.003])

    theta_vals = [-0.2, 0.1, 0, 0.1, 0.2]
    #R0_vals = [1.3,2.3,4]
    R0_vals = [2.3]
    migration_vals =  10**np.linspace(-0.25,2.75, 25)*1e-3
    eps_vals = np.linspace(0,0.8,17)
    amplitudes = []
    means = []
    costs = []
    for theta, R0, migration, eps in product(theta_vals, R0_vals, migration_vals, eps_vals):
        X= (R0*rec, eps, theta, rec, migration/frac_exposed, frac_exposed, turnover, incubation)
        traj = run_SIR(X, tmax, dt)
        num_points = len(traj)
        print(R0, migration, eps, np.mean(traj[num_points//2:,3]))
        # amplitudes.append(np.std(traj[num_points//2:,3])/np.mean(traj[num_points//2:,3]))
        amplitudes.append(np.max(traj[3*num_points//4:,3])/np.min(traj[3*num_points//4:,3]))
        means.append(np.mean(traj[num_points//2:,3]))
        costs.append(cost(traj, prevalence, dt=dt))


    amplitudes = np.reshape(amplitudes, (len(theta_vals), len(R0_vals), len(migration_vals), len(eps_vals)))
    means = np.reshape(means, (len(theta_vals), len(R0_vals), len(migration_vals), len(eps_vals)))
    costs = np.reshape(costs, (len(theta_vals), len(R0_vals), len(migration_vals), len(eps_vals)))

    for ri, R0 in enumerate(R0_vals):
        plt.figure()
        plt.title(f'log10(max/min) of incidence, R0={R0}')
        sns.heatmap(np.log10(amplitudes[2][ri]), xticklabels=[f"{x:1.1f}" for x in eps_vals],
                    yticklabels=[f"{x:1.1e}" for x in migration_vals])
        plt.tick_params('y', rotation=0)
        plt.ylabel('migration')
        plt.xlabel('seasonality')
        plt.tight_layout()
        plt.savefig(f"figures/oscillations_R0_{R0}_b_{turnover}.pdf")

    fs=16
    for ri, R0 in enumerate(R0_vals):
        fig, ax = plt.subplots()
        iax = ax.imshow(np.minimum(1/costs.min(axis=0)[ri][::-1],1),
                        interpolation='nearest', aspect='auto')
        cbar = fig.colorbar(iax) #, label='Goodness of fit (inverse squared distance)')
        cbar.ax.tick_params('y', labelsize=0.8*fs)

        plt.xticks(np.arange(0,len(eps_vals),2), [f"{x:1.1f}" for x in eps_vals[::2]])
        plt.yticks(np.arange(2,len(migration_vals), 4), [f"{x:1.1e}" for x in migration_vals[::-1][2::4]], rotation=0)
        plt.ylabel('migration', fontsize=fs)
        plt.xlabel('seasonality', fontsize=fs)
        plt.tick_params(labelsize=fs*0.8)
        plt.tight_layout()
        plt.savefig(f"figures/fit_R0_{R0}_b_{turnover}.pdf")


    for ri, R0 in enumerate(R0_vals):
        plt.figure()
        plt.imshow(costs.argmin(axis=0)[ri][::-1],
                        interpolation='nearest', aspect='auto')
                    #, xticklabels=[f"{x:1.1f}" for x in eps_vals],
                    #yticklabels=[f"{x:1.1e}" for x in migration_vals])

        plt.tick_params('y', rotation=0)
        plt.xticks(np.arange(0,len(eps_vals),2), [f"{x:1.1f}" for x in eps_vals[::2]])
        plt.yticks(np.arange(2,len(migration_vals), 4), [f"{x:1.1e}" for x in migration_vals[::-1][2::4]], rotation=0)
        plt.colorbar()
        plt.ylabel('migration')
        plt.xlabel('seasonality')
        plt.tight_layout()
        plt.savefig(f"figures/theta_R0_{R0}_b_{turnover}.pdf")

