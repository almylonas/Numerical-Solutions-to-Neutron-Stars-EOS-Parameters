import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.stats as stats
import seaborn as sns

mass=np.array([1.12,1.28,1.38,1.39,1.41,1.47,1.49,1.5,1.55,1.62,1.625,1.655,1.68,1.9,1.92,2.11,2.24])
names=np.array(["J0205+6449","J0537-6910","J1803-2137","J1826-1334","J1835-4510","J1932+2220","J1801-2451","J1420-6048","J1709-4429","J2021+3651","J1048-5832","J2229+6114","J730-3350","J1341-6220","J1105-6107","J0631+1036","J1413-6141"])
domega=np.array([3.63,2.65,2.25,2.22,2.17,1.94,1.89,1.86,1.76,1.57,1.55,1.49,1.44,1,0.97,0.72,0.53])
m=np.linspace(1,2.3,10000)
dosly=21.927*np.e**(-1.635*m)
dogm1=33.07*np.e**(-1.514*m)
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$\Delta\Omega(M)$")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"Maximum Glitch $\Delta\Omega [10^{-4}rad/s]$")
#plt.scatter(mass,domega,color='indianred',marker=".",s=50,label="$\Delta\Omega_{SLy}$")
plt.plot(m,dosly,color='indianred',linestyle="-",label="$\Delta\Omega_{SLy}$")
plt.plot(m,dogm1,color='navy',linestyle="-",label="$\Delta\Omega_{GM1}$")
plt.legend(loc="best")
plt.savefig("domega.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Depth of $F_{pin_{max}}$")
ax.set_xlabel(r"Stiffness of EoS")
ax.set_ylabel(r"Depth of $x(F_{pin_{max}})$ [km]")
radius=np.array([12.5,13.23,13.23,12.42,11.74,12.75,11.97,11.97,13.98,12.43,11.72,10.42,11.13,11.91,11.91,14.49,13.82])
pr=np.array([7.7063,8.0775,8.0775,7.6463,7.3313,7.8525,7.4438,7.4438,8.3475,7.6913,7.2938,6.5775,6.9750,7.3725,7.3725,8.6063,8.5088])
stiffness=np.array([-0.0385,0.0177,0.0177,-0.0446,-0.0969,-0.0192,-0.0792,-0.0792,0.0754,-0.0438,-0.0985,-0.1985,-0.1438,-0.0838,-0.0838,0.1146,0.061])
pinrad=radius-1.49895602413*pr
frac=pinrad/radius
z = np.polyfit(stiffness, frac, 1)
p = np.poly1d(z)
print("Trendline for Pinning depth/Radius-Stifness",p)
print("Pinning depth/Radius-Stifness:\n","Pearson: ", stats.pearsonr(stiffness, frac))
print("Spearman: ",stats.spearmanr(stiffness, frac))
print("Kendall: ",stats.kendalltau(stiffness, frac))
plt.scatter(stiffness,frac,s=100,color='navy',marker=".")
plt.plot(stiffness,p(stiffness),color='orchid')
plt.legend(loc="best")
plt.savefig("depthstiff.png")
plt.close()
d=np.linspace(0,18,1)
sns.jointplot(x =stiffness, y =frac, data =d, color ='purple', fill=True, kind ='kde')
plt.xlabel('Stiffness of EoS', fontsize=15)
plt.ylabel('$d(F_{pin_{max}})/R$', fontsize=15)