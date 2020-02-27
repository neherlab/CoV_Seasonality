import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

d = {}
viruses = ['HKU1_OC43', '229E', 'NL63']
for v in viruses:
    d[v] = pd.read_csv(f'data/CoV_{v}_by_month.tsv', sep='\t')

fs=16
for v in viruses:
    plt.figure()
    plt.title(v, fontsize=fs)
    jan = np.where(d[v]['month']==1)[0]
    for i,j in enumerate(jan):
        frac_pos =  d[v]['positive tests'][j:j+12]/(d[v]['positive tests'][j:j+12]+ d[v]['negative tests'][j:j+12])
        m = d[v]['month'][j:j+12]
        plt.plot(m, frac_pos/frac_pos.mean(), 'o-', lw=2, alpha=0.5, c=cm.viridis(i/10))

    plt.xlabel('Month', fontsize=fs)
    plt.ylabel('fraction positive (normalized)', fontsize=fs)
    plt.savefig(f'figures/fraction_positive_{v}.pdf')

for v in viruses:
    plt.figure()
    plt.title(v, fontsize=fs)
    jan = np.where(d[v]['month']==1)[0]
    for i,j in enumerate(jan):
        frac_pos =  d[v]['positive tests'][j:j+12]/(d[v]['positive tests'][j:j+12]+ d[v]['negative tests'][j:j+12])
        m = d[v]['month'][j:j+12]
        plt.plot(m, frac_pos, 'o-', lw=2, alpha=0.5, c=cm.viridis(i/10))

    plt.xlabel('Month', fontsize=fs)
    plt.ylabel('fraction positive', fontsize=fs)

for v in viruses:
    plt.figure()
    plt.title(v, fontsize=fs)
    jan = np.where(d[v]['month']==1)[0]
    for i,j in enumerate(jan):
        frac_pos =  d[v]['positive tests'][j:j+12]
        m = d[v]['month'][j:j+12]
        plt.plot(m, frac_pos, 'o-', lw=2, alpha=0.5, c=cm.viridis(i/10))

    plt.xlabel('Month', fontsize=fs)
    plt.ylabel('positive tests', fontsize=fs)