"""
script to run a simple seaonal SIR model with coupling to a global reservoir
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dSIRdt(tSI, beta, eps, theta, rec, influx, turnover):
    '''
    time derivative of a simple SIR model with an influx of infected
    (e.g. holiday returnees) and population turnover. Seasonality is
    modelled as harmonically oscillating infectivity with amplitude
    eps and peak at theta
    '''
    infection = beta*(1+eps*np.cos(2*np.pi*(tSI[0]-theta)))*tSI[1]*tSI[2]

    dS = - infection + (1-tSI[1])*turnover - influx
    dI = infection - (turnover+rec)*tSI[2] + influx
    return np.array([1, dS, dI])


def run_SIR(X, tmax, dt):
    '''
    generate a trajectory for parameters X of length tmax with integration
    time step dt
    '''
    beta, eps, theta, rec, influx, turnover = X

    # initialize trajectory at steady state.
    tSI = [np.array([0, rec/beta, turnover*(R0-1)/(R0*rec)])]

    while tSI[-1][0]<tmax:
        dSIR = dSIRdt(tSI[-1], beta, eps, theta, rec, influx, turnover)
        tSI.append(tSI[-1] + dSIR*dt)

    return np.array(tSI)

def cost(traj, prevalence, plot=False, dt=0.001):
    avg = np.mean(np.array([[traj[-int((y+(11.5-m)/12)/dt),2] for y in range(10)]
            for m in range(12)]), axis=1)

    C = np.sum((avg/avg.mean() - prevalence/prevalence.mean())**2)
    if plot:
        plt.plot(avg/avg.mean())
        plt.plot(prevalence/prevalence.mean())
    print(C)
    return C

if __name__ == '__main__':
    # read average seasonal test-positive rate.
    df = pd.read_csv('frac_positive_by_month.tsv', sep='\t')
    from itertools import product
    # all time units are years. A rate of 1/day is hence 365/year
    # 1/week is 52/year

    R0 = 3
    rec = 36
    eps = 0.5
    theta = -0.15
    influx = 1e-3
    turnover = 0.1 # rate at which people turn susceptible (0.1 corresponds to once every 10 years)
    dt=0.001
    tmax = 30

    X0 = (R0*rec, eps, theta, rec, influx, turnover)
    traj = run_SIR(X0, tmax, dt=dt)
    prevalence = df["all CoVs"]
    cost(traj, prevalence, dt=dt)
    plt.plot(traj[-3000:,0], traj[-3000:,2])
    plt.ylim([0,0.003])

    theta_vals = [-0.2, 0.1, 0, 0.1, 0.2]
    R0_vals = [1.5,2.5,4]
    influx_vals =  2**np.arange(0,10, 0.5)*1e-4
    eps_vals = np.linspace(0,0.8,17)
    amplitudes = []
    means = []
    costs = []
    for theta, R0, influx, eps in product(theta_vals, R0_vals, influx_vals, eps_vals):
        X= (R0*rec, eps, theta, rec, influx, turnover)
        traj = run_SIR(X, tmax, dt)
        num_points = len(traj)
        print(R0, influx, eps, np.mean(traj[num_points//2:,2]))
        # amplitudes.append(np.std(traj[num_points//2:,2])/np.mean(traj[num_points//2:,2]))
        amplitudes.append(np.max(traj[3*num_points//4:,2])/np.min(traj[3*num_points//4:,2]))
        means.append(np.mean(traj[num_points//2:,2]))
        costs.append(cost(traj, prevalence, dt=dt))


    amplitudes = np.reshape(amplitudes, (len(theta_vals), len(R0_vals), len(influx_vals), len(eps_vals)))
    means = np.reshape(means, (len(theta_vals), len(R0_vals), len(influx_vals), len(eps_vals)))
    costs = np.reshape(costs, (len(theta_vals), len(R0_vals), len(influx_vals), len(eps_vals)))

    for ri, R0 in enumerate(R0_vals):
        plt.figure()
        plt.title(f'log10(max/min) of incidence, R0={R0}')
        sns.heatmap(np.log10(amplitudes[2][ri]), xticklabels=[f"{x:1.1f}" for x in eps_vals],
                    yticklabels=[f"{x:1.1e}" for x in influx_vals])
        plt.tick_params('y', rotation=0)
        plt.ylabel('import rate')
        plt.xlabel('seasonality')
        plt.tight_layout()
        plt.savefig(f"figures/oscillations_{R0}.pdf")

    fs=16
    for ri, R0 in enumerate(R0_vals):
        fig, ax = plt.subplots()
        iax = ax.imshow(np.minimum(1/costs.min(axis=0)[ri][::-1],1),
                        interpolation='nearest', aspect='auto')
        cbar = fig.colorbar(iax) #, label='Goodness of fit (inverse squared distance)')
        cbar.ax.tick_params('y', labelsize=0.8*fs)

        plt.xticks(np.arange(0,len(eps_vals),2), [f"{x:1.1f}" for x in eps_vals[::2]])
        plt.yticks(np.arange(len(influx_vals)-1,-1,-2), [f"{x:1.1e}" for x in influx_vals[::2]], rotation=0)
        plt.ylabel('import rate', fontsize=fs)
        plt.xlabel('seasonality', fontsize=fs)
        plt.tick_params(labelsize=fs*0.8)
        plt.tight_layout()
        plt.savefig(f"figures/fit_{R0}.pdf")


    for ri, R0 in enumerate(R0_vals):
        plt.figure()
        sns.heatmap(costs.argmin(axis=0)[ri], xticklabels=[f"{x:1.1f}" for x in eps_vals],
                    yticklabels=[f"{x:1.1e}" for x in influx_vals])
        plt.tick_params('y', rotation=0)
        plt.ylabel('import rate')
        plt.xlabel('seasonality')
        plt.tight_layout()
        plt.savefig(f"figures/theta_{R0}.pdf")

