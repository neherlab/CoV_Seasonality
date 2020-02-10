import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dSIRdt(tSI, beta, eps, phi, rec, influx, turn_over):
    infection = beta*(1+eps*np.cos(2*np.pi*(tSI[0]+phi)))*tSI[1]*tSI[2]

    dS = - infection + (1-tSI[1])*turn_over - influx
    dI = infection - (turn_over+rec)*tSI[2] + influx
    return np.array([1, dS, dI])

def run_SIR(X, tmax, dt):
    turn_over = 0.03

    R0, eps, phi, rec, log10_influx = X
    influx = 10**log10_influx

    SI_vs_t = [np.array([0, 1/R0, 0.001])]
    while SI_vs_t[-1][0]<tmax:
        dSIR = dSIRdt(SI_vs_t[-1], R0*rec, eps, phi, rec, influx, turn_over)
        SI_vs_t.append(SI_vs_t[-1] + dSIR*dt)

    return np.array(SI_vs_t)

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

    df = pd.read_csv('frac_positive_by_month.tsv', sep='\t')
    R0 = 3
    rec = 52
    log10_influx = -3
    eps = 0.5
    X0 = (R0, eps, 1.0, rec, log10_influx)

    traj = run_SIR(X0, 30, 0.001)
    cost(X0, df["all"])
    plt.plot(traj[3000:,0], traj[3000:,2])
    plt.ylim([0,0.003])

    # from scipy.optimize import minimize

    # sol = minimize(cost, X0, args=np.array(df['all']), method="SLSQP",
    #         bounds=[(1,6),(0, 0.9), (0.3,1.7), (10,100), (-4,-1)])