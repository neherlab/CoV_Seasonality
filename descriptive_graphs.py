import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("corona_2010_2020_2.csv", sep=';')

df.rename(mapper={"Ålder":"age", "Provnummer":"sample_id",
                  "Provtagn datum":"test_date",
                  "Ankomstdatum":"admission_date",
                  "Pos/neg (1/0)":"pos/neg",
                  "Resultat (kvantitet)":"result",
                  "Analys":"test"},
          inplace=True, axis="columns")


# format columns
for col in ["test_date", "admission_date"]:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='ignore')

df.loc[df["age"]>150, "age"] = np.nan

# split into different tests
tests = np.unique(df["test"])
data_by_test = {}

data_by_test['CoV 229E'] = df.loc[df["test"]=='Coronavirus 229E-RNA']
data_by_test['CoV NL63'] = df.loc[df["test"]=='Coronavirus NL63-RNA']
data_by_test['CoV HKU1/OC43'] = df.loc[(df["test"]=='Coronavirus HKU1-RNA')\
                                               |(df["test"]=='Coronavirus OC43-, HKU1-RNA')\
                                               |(df["test"]=='Coronavirus OC43-RNA')]

# data_by_test = {t:df.loc[df["test"]==t,:] for t in tests}
data_by_test['all CoVs'] = df
number_of_tests = {t:{"total": len(d), "positive":np.sum(d["pos/neg"]==1.0)}
                   for t,d in data_by_test.items()}


# make age distributions
bins = np.arange(0,100,5)
bin_centers = 0.5*(bins[1:] + bins[:-1])

fs=16
plt.figure()
for t,d in data_by_test.items():
    pos = d["pos/neg"]==1

    y,x = np.histogram(d.loc[pos, "age"], bins=bins)
    plt.plot(bin_centers, y, label=t, lw=3 if t=='all CoVs' else 2)

plt.legend(fontsize=0.8*fs)
plt.tick_params(labelsize=0.8*fs)
plt.xlabel('age', fontsize=fs)
plt.ylabel('number positive tests', fontsize=fs)
plt.tight_layout()
plt.savefig('figures/pos_count_vs_age.pdf')

plt.figure()
for t,d in data_by_test.items():
    neg = d["pos/neg"]==0
    pos = d["pos/neg"]==1

    yn,xn = np.histogram(d.loc[neg, "age"], bins=bins)
    yp,xp = np.histogram(d.loc[pos, "age"], bins=bins)
    plt.plot(bin_centers, yp/yn, label=t)

plt.legend(fontsize=0.8*fs)
plt.tick_params(labelsize=0.8*fs)
plt.xlabel('age', fontsize=fs)
plt.ylabel('fraction of positive tests', fontsize=fs)
plt.tight_layout()
plt.savefig('figures/pos_frac_vs_age.pdf')


# plot through time
plt.figure()
for t,d in data_by_test.items():
    x = d.groupby(by=lambda x:(d.loc[x,'admission_date'].year,d.loc[x,'admission_date'].month)).sum()
    time_ax = [i[0]+(i[1]-0.5)/12 for i in x.index]
    plt.plot(time_ax, x["pos/neg"], label=t, lw=4 if t=='all CoVs' else 2)

plt.legend(fontsize=0.8*fs)
plt.tick_params(labelsize=0.8*fs)
plt.xlabel('time', fontsize=fs)
plt.ylabel('number positive tests', fontsize=fs)
plt.tight_layout()
plt.savefig('figures/pos_count_vs_time.pdf')

plt.figure()
for t,d in data_by_test.items():
    x = d.groupby(by=lambda x:(d.loc[x,'admission_date'].year,d.loc[x,'admission_date'].month)).mean()
    time_ax = [i[0]+(i[1]-0.5)/12 for i in x.index]
    plt.plot(time_ax, x["pos/neg"], label=t, lw=4 if t=='all CoVs' else 2)

plt.legend(fontsize=0.8*fs)
plt.tick_params(labelsize=0.8*fs)
plt.xlabel('time', fontsize=fs)
plt.ylabel('fraction positive tests', fontsize=fs)
plt.tight_layout()
plt.savefig('figures/pos_frac_vs_time.pdf')

plt.figure()
frac_positive = []
for t,d in data_by_test.items():
    x = d.groupby(by=lambda x:d.loc[x,'admission_date'].month).mean()
    frac_positive.append([t, list(x["pos/neg"])])
    time_ax = [i for i in x.index]
    plt.plot(time_ax, x["pos/neg"], 'o-', label=t, lw=4 if t=='all CoVs' else 2)

with open("frac_positive_by_month.tsv", 'w') as fh:
    fh.write("month\t"+'\t'.join(x[0] for x in frac_positive)+'\n')
    for m in range(12):
        fh.write(f"{m}\t"+'\t'.join(str(x[1][m]) for x in frac_positive)+'\n')


plt.legend(fontsize=0.8*fs)
plt.tick_params(labelsize=0.8*fs)
plt.xlabel('month', fontsize=fs)
plt.ylabel('fraction positive tests', fontsize=fs)
plt.tight_layout()
plt.savefig('figures/pos_frac_by_month.pdf')
