import numpy as np
import matplotlib.pyplot as plt
from compartment_model import trajectory
from scipy.stats import poisson
import seaborn as sns

month_lookup = {
0:'Jan',
1:'Feb',
2:'Mar',
3:'Apr',
4:'May',
5:'Jun',
6:'Jul',
7:'Aug',
8:'Sep',
9:'Oct',
10:'Nov',
11:'Dec'
}

if __name__ == '__main__':

    rec = 72   # recovery rate 1/2 weeks, results in an average 2weeks/R0 average seasonal interval.
    incubation = 5/365
    migration = 1e-2 # rate of moveing per year
    N0,N1 = 6e7,1e8
    eps_hubei = 0.4
    R0_hubei = 1.8
    containment_hubei = 0.5
    containment_NH = 0.5
    theta_hubei = 0.0

    R0_vals = np.linspace(1.5,3,8)
    theta_vals = np.array([10, 10.5, 11, 11.5, 0, 0.5, 1, 1.5, 2, 2.5])/12
    for eps in [0.15, 0.3, 0.5, 0.7]:
        ratio = []
        for R0 in R0_vals:
            for theta_NH in theta_vals:
                print(eps, R0,theta_NH)
                # initially fully susceptible with one case in Hubei, no cases in NH
                initial_population = np.array([[1, 0, 100/N0], [1,0, 0]])
                params = np.array([[N0, R0_hubei*rec, rec, eps_hubei, theta_hubei, 1, containment_hubei, migration,incubation],
                                   [N1, R0*rec, rec, eps, theta_NH, 1, containment_NH, migration, incubation]])

                n_pops = len(params)


                dt = 0.001
                t0 = 2019.8
                tmax = 2021.5
                t, populations = trajectory(initial_population, t0, tmax, dt, params,
                                            resampling_interval=0, turnover=0)
                spacing=30
                NH = np.array(populations)[:,1,2][::spacing]
                peaks = np.where((NH[1:-1]>NH[2:])&(NH[1:-1]>NH[:-2])&(NH[1:-1]>10000/N1))[0]
                if len(peaks)==2:
                    ratio.append(NH[peaks[0]]/NH[peaks[1]])
                elif len(peaks)==1:
                    print(t[peaks[0]*spacing])
                    if (t[peaks[0]*spacing])%1<(theta_NH + 0.5)%1:
                        ratio.append(1000)
                    else:
                        ratio.append(0.001)
                else:
                    # import ipdb; ipdb.set_trace()
                    print("ambiguous peaks")
                    ratio.append(np.nan)

        fs=16
        ratio = np.reshape(ratio, (len(R0_vals), len(theta_vals)))
        cbar_ticks = [-3,-2,-1,0,1,2,3]

        fig, ax = plt.subplots()
        # plt.title('Ratio of first to second peak', fontsize=fs)
        iax = ax.imshow(np.minimum(np.maximum(np.log10(ratio), -3), 3),
                        interpolation='nearest', cmap='plasma', aspect='auto')
        cbar = fig.colorbar(iax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels([r"$10^{"+str(x)+"}$" for x in cbar_ticks])
        cbar.ax.set_ylabel('Ratio of first to second peak', fontsize=fs)
        cbar.ax.tick_params('y', labelsize=0.8*fs)
        plt.yticks(np.arange(0,len(R0_vals),2),[f"{x:1.1f}" for x in R0_vals[::2]], rotation=0)
        plt.xticks(np.arange(0,len(theta_vals),2),[f"{month_lookup[int(x*12-0.5)]}" for x in theta_vals[1::2]])
        plt.ylabel(r"$\langle R_0\rangle $", fontsize=fs)
        plt.xlabel(r"peak transmission $\theta$", fontsize=fs)
        plt.ylim(-0.5,-0.5+len(R0_vals))
        plt.xlim(-0.5,-0.5+len(theta_vals))
        plt.tick_params(labelsize=fs*0.8)
        plt.tight_layout()

        plt.savefig(f"figures/peak_ratio_{eps}.pdf")
