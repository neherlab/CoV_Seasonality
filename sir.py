import numpy as np
import matplotlib.pyplot as plt

def dSIRdt(tSI, beta, eps, phi, rec, turn_over, influx):
    infection = beta*(1+eps*np.cos(2*np.pi*(tSI[0]+phi)))*tSI[1]*tSI[2]
    dS = - infection + (1-tSI[1])*turn_over - influx
    dI = infection - (turn_over+rec)*tSI[2] + influx
    return np.array([1, dS, dI])


if __name__ == '__main__':

    eps = 0.5
    dt = 0.001

    rec =  30    # 1/week
    turn_over = 0.1 # 0.1/year

    influx = 0.00001*50 # one in 10k people return sick from abroad per week


    for phi in [0, 0.25, 0.5, 0.75]:
        plt.figure()
        plt.title(f"delay between peak and introduction: {phi*12} month")
        for bi, beta in enumerate(np.array([1.5, 2,4])):
            SI_vs_t = [np.array([0, 1/beta, 0.001])]
            while SI_vs_t[-1][0]<30:
                dSIR = dSIRdt(SI_vs_t[-1], beta*rec, eps, phi, rec, turn_over, influx)
                SI_vs_t.append(SI_vs_t[-1] + dSIR*dt)


            SI_vs_t = np.array(SI_vs_t)
            # plt.plot(SI_vs_t[:,0], SI_vs_t[:,1], label='susceptible')
            plt.plot(SI_vs_t[:,0], SI_vs_t[:,2], alpha=0.5, lw=2, c=f"C{bi}", ls='--') #, label=f'endemic, R0 = {beta/52}')

            SI_vs_t = [np.array([30, 1, 0.000])]
            while SI_vs_t[-1][0]<35:
                dSIR = dSIRdt(SI_vs_t[-1], beta*rec, eps, phi, rec, turn_over, influx)
                SI_vs_t.append(SI_vs_t[-1] + dSIR*dt)


            SI_vs_t = np.array(SI_vs_t)
            # plt.plot(SI_vs_t[:,0], SI_vs_t[:,1], label='susceptible')
            plt.plot(SI_vs_t[:,0], SI_vs_t[:,2], label=f'nCoV, R0 = {beta}', lw=2, c=f"C{bi}")

        plt.show()
        plt.xlim([25,35])
        plt.ylim([1e-6,1])

        plt.yscale('log')
        plt.xlabel('year')
        plt.ylabel('prevalence')
        plt.legend()
        plt.savefig(f'figures/prevalance_{phi}.png')
