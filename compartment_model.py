import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def dSIRdt_vec(S, I, t, params):
    '''
    vectorized derivative of SIR models in multiple populations.

    S, I are vectors with one entry for each population
    t is time
    params is an array of [[N, beta, rec, eps, phi, climate, containment],
                           [N, beta, rec, eps, phi, climate, containment], ...]

    '''
    infection = params[:,1]*(1 - params[:,6]*I**3/(0.03**3+I**3))*S*I*(1+params[:,3]*np.cos(2*np.pi*(t - params[:,4])))
    dS = -infection
    dI = infection - params[:,2]*I

    return dS, dI



if __name__ == '__main__':

    rec = 25
    R0 = 2
    N0 = 1e6
    dt = 0.003
    tmax = 2
    eps0 = 0.5
    phi0 = 0.0
    migration = 2e-2
    t = 0

    n_pops = 100
    # add Hubei
    params = [[N0, R0*rec, rec, eps0, phi0, 1, 0.5]]
    populations = [[1, 1/N0]]

    for i in range(n_pops-1):
        tmp = np.random.random()
        if tmp<0.5:
            climate = 0 #tropical
            eps = np.random.random()*0.2
            phi = np.random.random()
        elif tmp<0.85:
            climate = 1 # northern
            eps = np.random.random()*0.8
            phi = np.random.normal(0,0.15)
        else:
            climate = -1
            eps = np.random.random()*0.8
            phi = np.random.normal(0.5,0.15)

        beta = np.random.normal(loc=2.5, scale=1)*rec
        N = np.random.lognormal(12,1)
        containment = np.random.random()*0.5
        populations.append([1, 0])
        params.append([N, beta, rec, eps, phi, climate, containment])

    params = np.array(params)
    populations = [np.array(populations)]
    t = [2019.8]

    tmax = 2021
    while t[-1]<tmax:
        dS, dI = dSIRdt_vec(populations[-1][:,0], populations[-1][:,1], t[-1], params)
        populations.append(populations[-1] + dt*np.array([dS,dI]).T)

        I_tot = (params[:,0]*populations[-1][:,1]).sum()
        populations[-1][:,1] += poisson.rvs(migration*I_tot/n_pops*dt*np.ones(n_pops))/params[:,0]
        t.append(t[-1]+dt)

    populations = np.array(populations)
    total_inf = (params[:,0]*populations[:,:,1]).sum(axis=1)

    fs=16
    plt.figure()
    plt.plot(t, total_inf, lw=3, label='Total')

    def get_color(pi):
        if pi==0:
            return 'C1'
        elif params[pi][5]==0:
            return 'C2'
        elif params[pi][5]==1:
            return 'C3'
        elif params[pi][5]==-1:
            return 'C4'
        else:
            return '#BBBBBB'

    label_set = set()
    def get_label(pi):
        if pi==0:
            label = 'Hubei'
        elif params[pi][5]==0:
            label = 'tropical'
        elif params[pi][5]==1:
            label = 'North'
        elif params[pi][5]==-1:
            label = 'South'
        if label in label_set:
            return ''
        else:
            label_set.add(label)
            return label

    for pi in range(n_pops):
        plt.plot(t, populations[:,pi,1]*params[pi, 0], lw=3 if pi==0 else 1,
                 c=get_color(pi), label=get_label(pi))

    plt.legend(fontsize=fs*0.8)
    plt.yscale('log')
    plt.ylabel('Cases', fontsize=fs)
    plt.xticks(np.array([2020, 2020.25, 2020.5, 2020.75, 2021, 2021.25]),
               ['2020-01', '2020-04', '2020-07', '2020-10', '2021-01', '2021-04'])
    plt.xlabel('Time', fontsize=fs)
    plt.tick_params(labelsize=0.8*fs)
    plt.ylim([0.1,total_inf[:].max()*2])
    plt.tight_layout()
    plt.savefig('seeding.png')

