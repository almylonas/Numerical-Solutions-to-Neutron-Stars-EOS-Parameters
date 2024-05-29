import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
data = {'TMA': 13.82,
        'TM1': 14.49,
        'SFHOY': 11.91,
        'SFHO': 11.91,
        'WFF2': 11.13,
        'WFF1': 10.42,
        'MPA1': 12.43,
        'H4': 13.98,
        'ENG':11.97,
        'ALF2': 12.75,
        'BSk20': 11.74,
        'DD2F': 12.42,
        'DD2': 13.23,
        'BHBLP': 13.23,
        'APR3': 11.97,
        'APRLD': 12.5,
        'SLY': 11.72,
        'FPS':10.82,
        'GM1':13.91}
group_data = list(data.values())
stiffness = np.array(group_data)/13-1
group_names = list(data.keys())
fig, ax = plt.subplots()
ax.set_title(r"Stiffiness of various EoS")
ax.set_xlabel(r"Stiffness value")
ax.set_ylabel(r"Equation of State")
m1 = 0.0384 #soft-intermd limit
m2 = -0.0384 #stiff-intermd limit
m4 = 0.15
m3= -0.25
plt.figure()
ax.axvspan(m1, m2, alpha=.4, color='orange',label="Intermediate")
ax.axvspan(m1, m3, alpha=.5, color='limegreen',label="Soft")
ax.axvspan(m2, m4, alpha=.5, color='tomato',label="Stiff")
plt.axvline(x = -0.0384, color = 'azure')
plt.axvline(x = 0.0384, color = 'azure')
ax.set_xlim([-0.22, 0.13])
ax.barh(group_names, stiffness)
plt.legend(loc="best")
plt.show()

mm=[2.024964882,2.158357442,2.34551081,2.227584628,2.004031502,2.176487163,2.009604768,2.444974992,1.978539006,2.096031021,2.045848478,2.202358532,2.234242641,2.071,2.043345217,1.971782034,2.09407014,1.7947279]
rr=[1.0177,0.9031,0.9208,0.9208,1.0631,0.8562,1.0754,0.9562,0.9162,0.8015,0.9162,1.1146,0.9615,0.9554,0.9015,0.9808,1.0177,0.8319]
cd=[6.43089E+14,9.24062E+14,8.32051E+14,8.61569E+14,6.10033E+14,1.04627E+15,6.29195E+14,7.33062E+14,9.28015E+14,1.20355E+15,9.24062E+14,5.44829E+14,8.07042E+14,8.56073E+14,9.86E+14,7.21021E+14,6.39075E+14]
z1 = np.polyfit(rr, mm, 1)
z2 = np.polyfit(rr[0:17], cd, 1)
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$M(\mathcal{S})$")
ax.set_xlabel(r"$\mathcal{S}$")
ax.set_ylabel(r"Mass $[M_{\odot}]$")
plt.scatter(rr,mm,s=100,color='navy',marker=".")
plt.plot(rr, p1(rr),color='navy')
print("Trendline for M-S",p1)
print("Mass and stiffness relation:\n","Pearson: ", stats.pearsonr(rr, mm))
print("Spearman: ",stats.spearmanr(rr, mm))
print("Kendall: ",stats.kendalltau(rr, mm))
print("Trendline for rho_c-S",p2)
plt.legend(loc="best")
plt.savefig("ms.png")
plt.close()
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$\rho_{c}(\mathcal{S})$")
ax.set_xlabel(r"$\mathcal{S}$")
ax.set_ylabel(r"Central Density $\rho_{c}$")
plt.scatter(rr[0:17],cd,s=100,color='indianred',marker=".")
plt.plot(rr, p2(rr),color='indianred')
plt.legend(loc="best")
plt.savefig("rs.png")
plt.close()
print("Central Density and stiffness relation: \n","Pearson: ",stats.pearsonr(rr[0:17], cd))
print("Spearman: ",stats.spearmanr(rr[0:17], cd))
print("Kendall: ",stats.kendalltau(rr[0:17], cd))
d=np.linspace(0,17,1)
sns.jointplot(x =rr[0:17], y =cd, data =d, color ='red', fill=True, kind ='kde')
plt.xlabel('Stiffness of EoS', fontsize=15)
plt.ylabel('Central density ρ$_{c}$', fontsize=15)
plt.close()
plt.figure(figsize=(15,15))
ax=plt.gca()
d=np.linspace(0,18,1)
sns.jointplot(x =rr, y =mm, data =d, color ='green', fill=True, kind ='kde')
plt.xlabel('Stiffness of EoS', fontsize=15)
plt.ylabel('$M_{max}$ $[M_{\odot}]$', fontsize=15)