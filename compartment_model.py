import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

def dSIRdt_vec(S, I, t, params):
    '''
    vectorized derivative of SIR models in multiple populations.

    S, I are vectors with one entry for each population
    t is time
    params is an array of [[N, beta, rec, eps, theta, climate, containment],
                           [N, beta, rec, eps, theta, climate, containment], ...]

    '''
    infection = params[:,1]*(1 - params[:,6]*I**3/(0.03**3+I**3))*S*I*(1+params[:,3]*np.cos(2*np.pi*(t - params[:,4])))
    dS = -infection
    dI = infection - params[:,2]*I

    return dS, dI



if __name__ == '__main__':

    rec = 36   # 10 day serial interval
    R0 = 2     # together with eps=0.5, this corresponds to 2.5 in winter
    N0 = 1e7   # size of Wuhan
    eps0 = 0.5 # seasonality
    theta0 = 0.0 # peak transmissibility in Dec/Jan
    migration = 2e-2 # rate of moveing per year
    sigma = 1 # standard devitation of population size lognormal

    #total number of populations
    n_pops = 1000

    # add Hubei population with parameters specified above
    #          population size, beta, rec, eps, theta, NH, containment
    params = [[N0, R0*rec, rec, eps0, theta0, 1, 0.5]]
    # initially fully susceptible with one case
    populations = [[1, 1/N0]]

    # construct other populations
    for i in range(n_pops-1):
        tmp = np.random.random() # draw NH, SH, tropical
        if tmp<0.5:
            climate = 0 #tropical
            eps = np.random.random()*0.2
            theta = np.random.random()
        elif tmp<0.85:
            climate = 1 # northern
            eps = np.random.random()*0.6
            theta = np.random.normal(0,0.15)  # peak in Dec/Jan
        else:
            climate = -1
            eps = np.random.random()*0.6
            theta = np.random.normal(0.5,0.15) # peak in June/July

        beta = np.random.normal(loc=2.5, scale=1)*rec
        N = np.random.lognormal(np.log(7e10/n_pops)-sigma**2/2,sigma)
        containment = np.random.random()*0.5
        # add initially uninfected population and parameters
        populations.append([1, 0])
        params.append([N, beta, rec, eps, theta, climate, containment])

    params = np.array(params)
    populations = [np.array(populations)]

    # start simulation
    t = [2019.8]
    dt = 0.001
    tmax = 2022
    while t[-1]<tmax:
        dS, dI = dSIRdt_vec(populations[-1][:,0], populations[-1][:,1], t[-1], params)
        populations.append(populations[-1] + dt*np.array([dS,dI]).T)

        I_tot = (params[:,0]*populations[-1][:,1]).sum()
        populations[-1][:,1] += poisson.rvs(migration*I_tot/n_pops*dt*np.ones(n_pops))/params[:,0]
        t.append(t[-1]+dt)

    populations = np.array(populations)
    # weigh each infection trajectory by its population size
    total_inf = (params[:,0]*populations[:,:,1]).sum(axis=1)


    #####################################################################
    ### plot figures
    #####################################################################
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

    fs=16
    plt.figure()
    plt.plot(t, total_inf, lw=3, label='Total')

    for pi in range(50):
        plt.plot(t, populations[:,pi,1]*params[pi, 0], lw=3 if pi==0 else 1.5,
                 c=get_color(pi), label=get_label(pi))

    plt.legend(fontsize=fs*0.8, loc=8, ncol=2)
    plt.yscale('log')
    plt.ylabel('Cases', fontsize=fs)
    plt.xticks(np.array([2020, 2020.5, 2021, 2021.5, 2022]),
               ['2020-01', '2020-04', '2020-07', '2020-10', '2021-01', '2021-04'])
    plt.tick_params(labelsize=0.8*fs)
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.ylim([0.1,total_inf[:].max()*2])
    plt.tight_layout()
    plt.savefig('figures/global.pdf')

