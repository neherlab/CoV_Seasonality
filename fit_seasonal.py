"""
script to run a simple seaonal SIR model with coupling to a global reservoir
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dSIRdt(tSI, beta, eps, phi, rec, influx, turnover):
    '''
    time derivative of a simple SIR model with an influx of infected
    (e.g. holiday returnees) and population turnover. Seasonality is
    modelled as harmonically oscillating infectivity with amplitude
    eps and peak at phi
    '''
    infection = beta*(1+eps*np.cos(2*np.pi*(tSI[0]-phi)))*tSI[1]*tSI[2]

    dS = - infection + (1-tSI[1])*turnover - influx
    dI = infection - (turnover+rec)*tSI[2] + influx
    return np.array([1, dS, dI])


def run_SIR(X, tmax, dt):
    '''
    generate a trajectory for parameters X of length tmax with integration
    time step dt
    '''
    beta, eps, phi, rec, influx, turnover = X

    # initialize trajectory at steady state.
    tSI = [np.array([0, rec/beta, turnover*(R0-1)/(R0*rec)])]

    while tSI[-1][0]<tmax:
        dSIR = dSIRdt(tSI[-1], beta, eps, phi, rec, influx, turnover)
        tSI.append(tSI[-1] + dSIR*dt)

    return np.array(tSI)

def cost(X, prevalence, plot=False):
    dt = 0.001
    traj = run_SIR(X, 30, dt)
    avg = np.mean(np.array([[traj[-int((y+(11.5-m)/12)/dt),2] for y in range(10)]
            for m in range(12)]), axis=1)

    C = np.sum((avg/avg.mean() - prevalence/prevalence.mean())**2)
    if plot:
        plt.plot(avg/avg.mean())
        plt.plot(prevalence/prevalence.mean())
    print(X, C)
    return C

if __name__ == '__main__':
    # read average seasonal test-positive rate.
    df = pd.read_csv('frac_positive_by_month.tsv', sep='\t')
    from itertools import product
    # all time units are years. A rate of 1/day is hence 365/year
    # 1/week is 52/year

    R0 = 3
    rec = 52
    influx = 1e-3
    eps = 0.5
    turnover = 0.1 # rate at which people turn susceptible (0.1 corresponds to once every 10 years)
    X0 = (R0, eps, 1.0, rec, influx, turnover)

    traj = run_SIR(X0, 30, 0.001)
    cost(X0, df["all CoVs"])
    plt.plot(traj[3000:,0], traj[3000:,2])
    plt.ylim([0,0.003])

    R0_vals = [1.5,2.5,4]
    influx_vals =  [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    eps_vals = np.linspace(0,0.8,9)
    amplitudes = []
    means = []
    for R0, influx, eps in product(R0_vals, influx_vals, eps_vals):
        X= (R0*rec, eps, 0.0, rec, influx, turnover)
        traj = run_SIR(X, 30, 0.001)
        num_points = len(traj)
        print(R0, influx, eps, np.mean(traj[num_points//2:,2]))
        # amplitudes.append(np.std(traj[num_points//2:,2])/np.mean(traj[num_points//2:,2]))
        amplitudes.append(np.max(traj[3*num_points//4:,2])/np.min(traj[3*num_points//4:,2]))
        means.append(np.mean(traj[num_points//2:,2]))


    amplitudes = np.reshape(amplitudes, (len(R0_vals), len(influx_vals), len(eps_vals)))
    means = np.reshape(means, (len(R0_vals), len(influx_vals), len(eps_vals)))

    for ri, R0 in enumerate(R0_vals):
        plt.figure()
        plt.title(f'log10(max/min) of incidence, R0={R0}')
        sns.heatmap(np.log10(amplitudes[ri]), xticklabels=[f"{x:1.1f}" for x in eps_vals],
                    yticklabels=[f"{x:1.1e}" for x in influx_vals])
        plt.tick_params('y', rotation=0)
        plt.ylabel('import rate')
        plt.xlabel('seasonality')
        plt.tight_layout()
        plt.savefig(f"figures/oscillations_{R0}.pdf")


    # from scipy.optimize import minimize

    # sol = minimize(cost, X0, args=np.array(df['all']), method="SLSQP",
    #         bounds=[(1,6),(0, 0.9), (0.3,1.7), (10,100), (-4,-1)])