import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def dSIRdt(tSI, beta, rec, eps=0, phi=0):
    infection = beta*tSI[1]*tSI[2]*(1+eps*np.cos(2*np.pi*(tSI[0] + phi)))
    dS = -infection
    dI = infection - rec*tSI[2]
    return np.array([1, dS, dI])


def run_SIR(N, R0, t0, tmax, dt):
    SI_vs_t = [np.array([t0, 0, 1/N])]
    while SI_vs_t[-1][0]<tmax:
        dSIR = dSIRdt(SI_vs_t[-1], R0*rec, eps, phi, rec, influx, turn_over)
        SI_vs_t.append(SI_vs_t[-1] + dSIR*dt)

    return np.array(SI_vs_t)

if __name__ == '__main__':

    rec = 25
    R0 = 2
    N0 = 1e6
    dt = 0.003
    tmax = 2
    eps0 = 0.5
    migration = 2e-2
    t = 0

    containment_R0 = 0.5

    total_inf=[[t,1,N0]]
    pops = [[[R0*rec, N0, eps0], [np.array([0, 1, 1/N0])]]]
    while t<tmax:
        Itmp = 0
        Ntot = 0
        new_pops = []
        for params, traj in pops:
            Ntot += params[1]
            dSIR = dSIRdt(traj[-1], params[0]*(1-containment_R0*(traj[-1][-1]>0.03)), rec, eps=params[2])
            new_state = traj[-1] + dSIR*dt
            new_state[-1] += (total_inf[-1][1]*params[1]/total_inf[-1][2]**2-traj[-1][-1])*migration*0.1
            traj.append(new_state)
            Itmp += traj[-1][-1]*params[1]

            migrants = poisson.rvs(traj[-1][-1]*params[1]*migration*dt) if len(pops)<1000 else 0
            for m in range(migrants):
                beta = np.random.normal(loc=2.5, scale=1)*rec
                N = np.random.lognormal(12,1)
                eps = np.random.random()*0.8
                new_pops.append([[beta, N, eps], [np.array([t, 1, 1/N])]])
        pops.extend(new_pops)
        t+=dt
        total_inf.append([t, Itmp, Ntot])
        if Itmp>1e10:
            break

    total_inf = np.array(total_inf)
    fs=16
    plt.figure()
    plt.plot(total_inf[:,0], total_inf[:,1], lw=3, label='Total')

    labels = {0:"Hubei", 1:"other"}
    for pi, (params, traj) in enumerate(pops):
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,2]*params[1], lw=3 if pi==0 else 1,
                 c='C1' if pi==0 else '#BBBBBB', label=labels.get(pi,''))

    plt.legend(fontsize=fs*0.8)
    plt.yscale('log')
    plt.ylabel('Cases', fontsize=fs)
    plt.xticks(np.array([2020, 2020.25, 2020.5, 2020.75, 2021, 2021.25]) - 2020 + 0.2,
               ['2020-01', '2020-04', '2020-07', '2020-10', '2021-01', '2021-04'])
    plt.xlabel('Time', fontsize=fs)
    plt.tick_params(labelsize=0.8*fs)
    plt.xlim(0,1.4)
    plt.ylim([0.1,total_inf[:,1].max()*2])
    plt.tight_layout()
    plt.savefig('seeding.png')

