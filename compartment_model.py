import pandas as pd
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def dSIRdt_vec(S, E, I, t, params, turnover=0):
    '''
    vectorized derivative of SIR models in multiple populations.

    S, I are vectors with one entry for each population
    t is time
    params is an array of [[N, beta, rec, eps, theta, climate, containment, migration_rate, incubation_time],
                           [N, beta, rec, eps, theta, climate, containment, migration_rate, incubation_time], ...]

    '''
    infection = params[:,1]*(1 - params[:,6]*I**3/(0.03**3+I**3))*S*I*(1+params[:,3]*np.cos(2*np.pi*(t - params[:,4])))
    dS = -infection - turnover*(S-1) - params[:,7]*S
    dE = infection - (turnover+1.0/params[:,8]+params[:,7])*E
    dI = E/params[:,8] - (params[:,2]+turnover+params[:,7])*I

    return dS, dE, dI


def migrate(pop, params, dt):
    # migration
    n_pops = params.shape[0]
    for i in range(pop.shape[1]):
        N_tot = (params[:,0]*pop[:,i]).sum()
        pop[:,i] += poisson.rvs(N_tot/n_pops*dt*params[:,7])/params[:,0]


def resample(pop, params, overdispersion=1):
    return (poisson.rvs(pop.T*params[:,0]/overdispersion)/params[:,0]*overdispersion).T


def trajectory(initial_pops, t0, tmax, dt, params, resampling_interval=1/52, turnover=0):
    populations = [initial_pops]
    t = [t0]
    last_resample = t[-1]
    while t[-1]<tmax:
        dS, dE, dI = dSIRdt_vec(populations[-1][:,0], populations[-1][:,1],populations[-1][:,2], t[-1],
                                params, turnover=turnover)
        populations.append(populations[-1] + dt*np.array([dS, dE, dI]).T)
        migrate(populations[-1], params, dt)
        if resampling_interval>0 and t[-1]-last_resample>resampling_interval:
            populations[-1] = resample(populations[-1], params, overdispersion=10)
            last_resample = t[-1]
        t.append(t[-1]+dt)

    return t, np.array(populations)


def plot_many_population_scenario(R0=2, t0=2019.9, tmax=2022, eps_temperate=0.5, eps_tropical=0.2,
                                  R0_sigma=0.5, population_turnover=0.1, containment_hubei=0.5,
                                  containment_world_range=0.5, plot_three_panel=True):
    # R0 = 2     # together with eps0=0.5, this corresponds to 3.0 in winter
    # containment_hubei = 0.5 # containment value for hubei. Strong containment measures in Hubei
    # containment_world_range = 0.5 # assumes uniform distribution between no containment and 0.5 in other regions

    rec = 72   # recovery rate 1/2 weeks, results in an average 2weeks/R0 average seasonal interval.
    incubation = 5/365
    N0 = 6e7   # size of Hubei
    eps0 = 0.4 # seasonality in Hubei
    theta0 = 0.0 # peak transmissibility in Dec/Jan
    popsize_sigma = 1 # standard deviation of population size lognormal
    world_population = 7.6e9
    migration = 1e-2 # rate of moving per year anywhere
    hubei_migration = 3e-3 # rate of moving per year anywhere
    migration_sigma = 1 # standard deviation migration rate lognormal
    theta_temperate_sigma = 0.1  # standard deviation of the distribution of peak x-missibility in temperate regions
    regions = {"Northern temperate":1, "Tropical":0,  "Southern temperate":-1}
    under_reporting = 3
    dt=0.001

    #total number of populations
    n_pops = 1000
    case_counts = pd.read_csv('data/case_counts.tsv', sep='\t')
    # add Hubei population with parameters specified above
    #          population size, beta, rec, eps, theta, NH, containment, relative migration
    params = [[N0, R0*rec, rec, eps0, theta0, 1, containment_hubei, hubei_migration, incubation]]

    # initially fully susceptible with one case
    initial_pops = [[1, 0, 30/N0]]

    # construct other initial_pops
    for i in range(n_pops-1):
        tmp = np.random.random() # draw NH, SH, tropical
        if tmp<0.4: # according to http://worldpopulationreview.com/countries/tropical-countries/
            climate = regions["Tropical"]
            eps = np.random.random()*eps_tropical
            theta = np.random.random()
        elif tmp<0.9: # 90% live in the northern hemisphere
            climate = regions["Northern temperate"]
            eps = 0.25+np.random.random()*eps_temperate
            theta = np.random.normal(0,theta_temperate_sigma)  # peak in Dec/Jan
        else:
            climate = regions["Southern temperate"]
            eps = np.random.random()*eps_temperate
            theta = np.random.normal(0.5,theta_temperate_sigma) # peak in June/July

        # Limit R0 to have to be >= 0
        beta = max(0, np.random.normal(loc=R0, scale=R0_sigma))*rec
        migration_rate = np.random.lognormal(-migration_sigma**2/2, migration_sigma)*migration
        N = np.random.lognormal(np.log(world_population/n_pops)-popsize_sigma**2/2,popsize_sigma)
        containment = np.random.random()*containment_world_range
        # add initially uninfected population and parameters
        initial_pops.append([1, 0, 0])
        params.append([N, beta, rec, eps, theta, climate, containment, migration_rate, incubation])

    params = np.array(params)
    # start simulation
    t, populations = trajectory(np.array(initial_pops), t0, tmax, dt, params,
                                resampling_interval=1/52, turnover=population_turnover)
    # weigh each infection trajectory by its population size
    total_inf = (params[:,0]*populations[:,:,2]).sum(axis=1)
    params_by_region = {r:params[params[:,5]==regions[r],:] for r in regions}
    pops_by_region = {r:populations[:,params[:,5]==regions[r],:] for r in regions}


    #####################################################################
    ### plot figures
    #####################################################################
    colors = {"Northern temperate":"C1", "Southern temperate":"C2", "Tropical":"C3", "Hubei":"C4", "Total simulation":"C0", "Total observed":"C5"}

    label_set = set()
    def get_label(pi):
        if pi==0:
            label = 'Hubei'
        elif params[pi][5]==0:
            label = 'Tropical'
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
    if plot_three_panel:
        # keep track of the lines we plot, so we can make one legend for whole figure
        subLines = {}

        fig, axs = plt.subplots(1, 3, figsize=(14,5), sharey=True)
        for ax in axs:
            totalLine = ax.plot(t, total_inf, lw=3, label='Total', c=colors["Total simulation"])
            observedLine = ax.plot(case_counts["date"], case_counts["total"]*under_reporting,
                    lw=3, label=f'Observed (x{under_reporting})', c=colors[f"Total observed"])
            subLines['Total'] = totalLine[0]
            subLines[f'Observed (x{under_reporting})'] = observedLine[0]

        #find max R0 to adjust colour shading when plotting lines
        maxR0 = max(params[:,1]/rec)

        for ax,r in zip(axs, regions.keys()):
            ax.set_title(r, fontsize=fs*1.2)
            subLine = ax.plot(t, (params_by_region[r][:,0]*pops_by_region[r][:,:,2]).sum(axis=1), lw=3, label="Sub-total", c=colors[r])
            subLines["Sub-total {}".format(r)] = subLine[0]
            label_set = set()
            if r=='Northern temperate':
                hubeiLine = ax.plot(t, populations[:,0,2]*params[0, 0], lw=3, c=colors["Hubei"], label="Hubei")
                subLines['Hubei'] = hubeiLine[0]

            for pi in range(min(30, len(params_by_region[r]))):
                if not(r=='Northern temperate' and pi == 0): #don't plot Hubei twice...
                    ax.plot(t, pops_by_region[r][:,pi,2]*params_by_region[r][pi, 0], lw=1.5, c=colors[r],
                        alpha=params_by_region[r][pi,1]/rec/maxR0)
                    label_set.add(r)

            # make custom legend to show R0 values
            custom_lines = [Line2D([0], [0], color=colors[r], lw=2, alpha=(1/maxR0)),
                            Line2D([0], [0], color=colors[r], lw=2, alpha=(R0/maxR0)),
                            Line2D([0], [0], color=colors[r], lw=2, alpha=1)]
            first_legend = ax.legend(custom_lines, ["1",R0,round(maxR0,1)], title="R0", fontsize=fs*0.8, loc=4)#4)
            plt.setp(first_legend.get_title(),fontsize=fs)
            ax.add_artist(first_legend)

            # dont plot indiv leg anymore, use one leg at bottom
            #ax.legend(fontsize=fs*0.8, loc=8, ncol=1)
            ax.set_yscale('log')
            if ax==axs[0]:
                ax.set_ylabel('Cases', fontsize=fs)
            ax.set_xticks(np.array([2020.0, 2020.5, 2021.0, 2021.5, 2022.0]))
            ax.set_xticklabels(['2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01', '2022-01-01'], horizontalalignment='right')
            ax.tick_params(axis='x', labelsize=0.8*fs, labelrotation=30)
            ax.tick_params(axis='y', labelsize=0.8*fs)
            ax.set_ylim([1,total_inf[:].max()*2])

        #plot 'observed' again now, so it's on top, and more visible (as such short line)
        for ax in axs:
            #make a lighter 'shadow' around the line so it's more visible
            ax.plot(case_counts["date"], case_counts["total"]*under_reporting,
                    lw=4, label=f'Observed (x{under_reporting})', c='white')
            ax.plot(case_counts["date"], case_counts["total"]*under_reporting,
                    lw=3, label=f'Observed (x{under_reporting})', c=colors[f"Total observed"])

        fig.legend(subLines.values(), subLines.keys(), loc='lower center', ncol=3, fontsize=fs*0.8)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.27)
        plt.savefig(f'figures/global_3_panel_{R0}.pdf')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.plot(t, total_inf, lw=3, label='Total', c=colors["Total simulation"])

        for r in regions:
            ax.plot(t, (params_by_region[r][:,0]*pops_by_region[r][:,:,2]).sum(axis=1), lw=3, label="Sub-total "+ r, c=colors[r])

        ax.legend(fontsize=fs*0.8, loc=1, ncol=1)
        ax.set_yscale('log')
        ax.set_ylabel('Cases', fontsize=fs)
        ax.tick_params(axis='x', labelsize=0.8*fs)
        ax.tick_params(axis='y', labelsize=0.8*fs)
        ax.set_ylim([1000,total_inf[:].max()*2])

        plt.tight_layout()
        plt.savefig(f'figures/global_endemic_{R0}.pdf')


if __name__ == '__main__':
    t0_vec = [2019.8, 2019.95, 2020.0]
    R0_vec = [1.5, 2.2, 3.0]

    for t0, R0 in zip(t0_vec, R0_vec):
        plot_many_population_scenario(R0=R0, t0=t0, eps_temperate=0.5, R0_sigma=0.5, tmax=2022,
                                      population_turnover=0.0, containment_hubei=0.5)

<<<<<<< HEAD
    for t0, R0 in zip([2019.6, 2019.8, 2020], [1.4, 1.8, 2.7]):
=======
    for t0, R0 in zip(t0_vec, R0_vec):
>>>>>>> SEIR
        plot_many_population_scenario(R0=R0, t0=t0, eps_temperate=0.5, R0_sigma=0.5,
                                      tmax=2032, population_turnover=0.1, plot_three_panel=False)
