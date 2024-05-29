import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
import matplotlib.colors as mcolors
import matplotlib.cm as cm
#from scipy.interpolate import CubicSpline

columns=["r","m","nu","rho","P"]

#reading the profiles

sly=pd.read_csv("./profiles/sly.csv", usecols=columns)
aprld=pd.read_csv("./profiles/aprld.csv", usecols=columns)
apr3=pd.read_csv("./profiles/apr3.csv", usecols=columns)
bhblp=pd.read_csv("./profiles/bhblp.csv", usecols=columns)
dd2f=pd.read_csv("./profiles/dd2f.csv", usecols=columns)
dd2=pd.read_csv("./profiles/dd2.csv", usecols=columns)
eosAU=pd.read_csv("./profiles/eosAU.csv", usecols=columns)
eosUU=pd.read_csv("./profiles/eosUU.csv", usecols=columns)
bsk=pd.read_csv("./profiles/bsk20.csv", usecols=columns)
ls220=pd.read_csv("./profiles/ls220.csv", usecols=columns)
ls375=pd.read_csv("./profiles/ls375.csv", usecols=columns)
alf2=pd.read_csv("./profiles/alf2.csv", usecols=columns)
eng=pd.read_csv("./profiles/eng.csv", usecols=columns)
gnh3=pd.read_csv("./profiles/gnh3.csv", usecols=columns)
h4=pd.read_csv("./profiles/h4.csv", usecols=columns)
mpa1=pd.read_csv("./profiles/mpa1.csv", usecols=columns)
wff1=pd.read_csv("./profiles/wff1.csv", usecols=columns)
wff2=pd.read_csv("./profiles/wff2.csv", usecols=columns)
sfho=pd.read_csv("./profiles/sfho.csv", usecols=columns)
sfhoy=pd.read_csv("./profiles/sfhoy.csv", usecols=columns)
sfhx=pd.read_csv("./profiles/sfhx.csv", usecols=columns)
tm1=pd.read_csv("./profiles/tm1.csv", usecols=columns)
tma=pd.read_csv("./profiles/tma.csv", usecols=columns)

rhotimes=6.1752364e+17
rtimes=1.49895602413
ptimes=5.6176471e+38

#Final densities
slyrho=sly.rho*rhotimes
aprldrho=aprld.rho*rhotimes
apr3rho=apr3.rho*rhotimes
bhblprho=bhblp.rho*rhotimes
dd2frho=dd2f.rho*rhotimes
dd2rho=dd2.rho*rhotimes
eosAUrho=eosAU.rho*rhotimes
eosUUrho=eosUU.rho*rhotimes
bskrho=bsk.rho*rhotimes
ls220rho=ls220.rho*rhotimes
ls375rho=ls375.rho*rhotimes
alf2rho=alf2.rho*rhotimes
engrho=eng.rho*rhotimes
h4rho=h4.rho*rhotimes
mpa1rho=mpa1.rho*rhotimes
wff1rho=wff1.rho*rhotimes
wff2rho=wff2.rho*rhotimes
sfhorho=sfho.rho*rhotimes
sfhoyrho=sfhoy.rho*rhotimes
sfhxrho=sfhx.rho*rhotimes
tm1rho=tm1.rho*rhotimes
tmarho=tma.rho*rhotimes
gnh3rho=gnh3.rho*rhotimes

#Final distances
slyr=sly.r*rtimes
aprldr=aprld.r*rtimes
apr3r=apr3.r*rtimes
bhblpr=bhblp.r*rtimes
dd2fr=dd2f.r*rtimes
dd2r=dd2.r*rtimes
eosAUr=eosAU.r*rtimes
eosUUr=eosUU.r*rtimes
bskr=bsk.r*rtimes
ls220r=ls220.r*rtimes
ls375r=ls375.r*rtimes
alf2r=alf2.r*rtimes
engr=eng.r*rtimes
h4r=h4.r*rtimes
mpa1r=mpa1.r*rtimes
wff1r=wff1.r*rtimes
wff2r=wff2.r*rtimes
sfhor=sfho.r*rtimes
sfhoyr=sfhoy.r*rtimes
sfhxr=sfhx.r*rtimes
tm1r=tm1.r*rtimes
tmar=tma.r*rtimes
gnh3r=gnh3.r*rtimes

#Final pressures
slyp=sly.P*ptimes
aprldp=aprld.P*ptimes
apr3p=apr3.P*ptimes
bhblpp=bhblp.P*ptimes
dd2fp=dd2f.P*ptimes
dd2p=dd2.P*ptimes
eosAUp=eosAU.P*ptimes
eosUUp=eosUU.P*ptimes
bskp=bsk.P*ptimes
ls220p=ls220.P*ptimes
ls375p=ls375.P*ptimes
alf2p=alf2.P*ptimes
engp=eng.P*ptimes
h4p=h4.P*ptimes
mpa1p=mpa1.P*ptimes
wff1p=wff1.P*ptimes
wff2p=wff2.P*ptimes
sfhop=sfho.P*ptimes
sfhoyp=sfhoy.P*ptimes
sfhxp=sfhx.P*ptimes
tm1p=tm1.P*ptimes
tmap=tma.P*ptimes
gnh3p=gnh3.P*ptimes


plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Density profiles")
ax.set_xlabel(r"Distance from the center $r$ $[km]$")
ax.set_ylabel(r"Density $\rho(r)$ $[g/cm_{3}]$")
plt.plot(dd2r,dd2rho, color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(bskr,bskrho,color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(apr3r,apr3rho,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(engr,engrho,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(tmar,tmarho,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(wff2r,wff2rho,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(h4r,h4rho,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(ls375r,ls375rho,color='orange', linestyle='dashed',label="LS375 EoS")
plt.plot(mpa1r,mpa1rho,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(sfhoyr,sfhoyrho,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(ls220r,ls220rho,color='darkkhaki',linestyle='dashed',label="LS220 EoS")
plt.plot(wff1r,wff1rho,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(sfhor,sfhorho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(eosAUr,eosAUrho,color='darksalmon',linestyle='dashed',label="eosAU EoS")
plt.plot(sfhxr,sfhxrho,color='orchid',linestyle='dashed',label="SFHX EoS")
plt.plot(tm1r,tm1rho,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(aprldr,aprldrho,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(bhblpr,bhblprho,color='steelblue',linestyle='dashed',label="BHBL[ EoS")
plt.plot(dd2fr,dd2frho,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(eosUUr,eosUUrho,color='dodgerblue',linestyle='dashed',label="eosUU EoS")
plt.plot(slyr,slyrho,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(alf2r,alf2rho,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.plot(gnh3r,gnh3rho,color='teal',linestyle='dashed',label="GNH3 EoS")
plt.legend(loc="best")
plt.savefig("densities.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Mass profiles")
ax.set_xlabel(r"Distance from the center $r$ $[km]$")
ax.set_ylabel(r"Mass $m(r)$ $[M\odot]$")
plt.plot(dd2r,dd2.m, color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(bskr,bsk.m,color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(apr3r,apr3.m,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(engr,eng.m,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(tmar,tma.m,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(wff2r,wff2.m,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(h4r,h4.m,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(ls375r,ls375.m,color='orange', linestyle='dashed',label="LS375 EoS")
plt.plot(mpa1r,mpa1.m,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(sfhoyr,sfhoy.m,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(ls220r,ls220.m,color='darkkhaki',linestyle='dashed',label="LS220 EoS")
plt.plot(wff1r,wff1.m,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(sfhor,sfho.m,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(eosAUr,eosAU.m,color='darksalmon',linestyle='dashed',label="eosAU EoS")
plt.plot(sfhxr,sfhx.m,color='orchid',linestyle='dashed',label="SFHX EoS")
plt.plot(tm1r,tm1.m,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(aprldr,aprld.m,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(bhblpr,bhblp.m,color='steelblue',linestyle='dashed',label="BHBL[ EoS")
plt.plot(dd2fr,dd2f.m,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(eosUUr,eosUU.m,color='dodgerblue',linestyle='dashed',label="eosUU EoS")
plt.plot(slyr,sly.m,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(alf2r,alf2.m,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.plot(gnh3r,gnh3.m,color='teal',linestyle='dashed',label="GNH3 EoS")
plt.legend(loc="best")
plt.savefig("masses.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Pressure profiles")
ax.set_xlabel(r"Distance from the center $r$ $[km]$")
ax.set_ylabel(r"Pressure $P(r)$ $[dyn*cm^{-2}]$")
plt.plot(dd2r,dd2p, color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(bskr,bskp,color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(apr3r,apr3p,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(engr,engp,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(tmar,tmap,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(wff2r,wff2p,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(h4r,h4p,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(ls375r,ls375p,color='orange', linestyle='dashed',label="LS375 EoS")
plt.plot(mpa1r,mpa1p,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(sfhoyr,sfhoyp,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(ls220r,ls220p,color='darkkhaki',linestyle='dashed',label="LS220 EoS")
plt.plot(wff1r,wff1p,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(sfhor,sfhop,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(eosAUr,eosAUp,color='darksalmon',linestyle='dashed',label="eosAU EoS")
plt.plot(sfhxr,sfhxp,color='orchid',linestyle='dashed',label="SFHX EoS")
plt.plot(tm1r,tm1p,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(aprldr,aprldp,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(bhblpr,bhblpp,color='steelblue',linestyle='dashed',label="BHBL[ EoS")
plt.plot(dd2fr,dd2fp,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(eosUUr,eosUUp,color='dodgerblue',linestyle='dashed',label="eosUU EoS")
plt.plot(slyr,slyp,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(alf2r,alf2p,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.plot(gnh3r,gnh3p,color='teal',linestyle='dashed',label="GNH3 EoS")
plt.legend(loc="best")
plt.savefig("pressures.png")
plt.close()

cd=[dd2rho[0],bskrho[0],apr3rho[0],engrho[0],tmarho[0],wff2rho[0],wff1rho[0],h4rho[0],mpa1rho[0],sfhoyrho[0],sfhorho[0],tm1rho[0],aprldrho[0],bhblprho[0],dd2frho[0],slyrho[0],alf2rho[0]]
print(cd)
#Polynomial density profile approximation
r=np.linspace(0, 12,10000)
def density(r,n):
    Ms=2*10**33
    masses=1.4
    M=Ms*masses
    c=1
    R=1.2*10**6
    return ((15*M*c**2)/(8*np.pi*R**3))*(1-((r/(R/10**5))**n))
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Approximate Density profiles")
ax.set_xlabel(r"Distance from the center $r$ $[km]$")
ax.set_ylabel(r"Density $\rho(r)$ $[g/cm_{3}]$")
plt.plot(r,density(r,1.6),color='forestgreen',linestyle='dashed',label="n=1.6")
plt.plot(r,density(r,1.8),color='purple',linestyle='dashed',label="n=1.8")
plt.plot(r,density(r,2),color='navy',linestyle='dashed',label="n=2")
plt.plot(r,density(r,2.2),color='firebrick',linestyle='dashed',label="n=2.2")
plt.plot(r,density(r,2.4),color='cyan',linestyle='dashed',label="n=2.4")
plt.plot(r,density(r,2.6),color='hotpink',linestyle='dashed',label="n=2.6")
plt.plot(r,density(r,2.8),color='teal', linestyle='dashed',label="n=2.8")
plt.plot(r,density(r,3),color='peru', linestyle='dashed',label="n=3")
plt.legend(loc="best")
plt.savefig("denapprox.png")
plt.close()

#Crust Inertia
masses=np.array([1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3])
r_sly=np.array([11.86,11.83,11.80,11.76,11.71,11.64,11.55,11.42,11.26,11.03,10.62])
r_gm1=np.array([13.94,13.94,13.94,13.93,13.91,13.89,13.85,13.79,13.72,13.62,13.49,13.33,13.1,12.71])
rc_sly=np.array([10.35,10.49,10.6,10.69,10.75,10.79,10.79,10.76,10.68,10.54,10.23])
ric_sly=np.array([11.23,11.28,11.31,11.32,11.32,11.29,11.24,11.16,11.03,10.83,10.47])
rc_gm1=np.array([11.79,12.01,12.19,12.35,12.47,12.58,12.66,12.71,12.74,12.74,12.7,12.63,12.48,12.2])
ric_gm1=np.array([13.02,13.12,13.20,13.27,13.32,13.34,13.35,13.35,13.32,13.26,13.17,13.05,12.85,12.51])
sly_inertia=np.array([0.739,0.827,0.914,0.999,1.079,1.154,1.222,1.279,1.322,1.339,1.299])*10**45
gm1_inertia=np.array([1.021,1.146,1.271,1.395,1.516,1.634,1.747,1.854,1.954,2.043,2.118,2.173,2.194,2.146])*10**45
sly_c_inertia=np.array([0.697,0.788,0.878,0.965,1.048,1.126,1.197,1.258,1.303,1.324,1.289])*10**45
gm1_c_inertia=np.array([0.896,1.025,1.156,1.285,1.412,1.536,1.657,1.771,1.878,1.974,2.057,2.120,2.15,2.113])*10**45
sly_cr_inertia=sly_inertia-sly_c_inertia
gm1_cr_inertia=gm1_inertia-gm1_c_inertia
def percent(total,crust):
    return crust/total
def crust(total,core):
    return total-core
def inner(core,outer):
    return outer-core
def outer(outer,total):
    return total-outer
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$I_{crust}/I_{total}$")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"$I_{crust}/I_{total}$")
plt.plot(masses[0:11],percent(sly_inertia,sly_cr_inertia),color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,percent(gm1_inertia,gm1_cr_inertia),color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("percent.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$I_{crust}$")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"$I_{crust}$")
plt.plot(masses[0:11],sly_cr_inertia,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,gm1_cr_inertia,color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("inertiacrust.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"$I_{core}$")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"$I_{crust}$")
plt.plot(masses[0:11],sly_c_inertia,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,gm1_c_inertia,color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("inertiacore.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Crust Thickness")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"Crust thickness $[km]$")
plt.plot(masses[0:11],crust(r_sly,rc_sly),color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,crust(r_gm1,rc_gm1),color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("coreradius.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Inner Crust Thickness")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"Inner crust thickness $[km]$")
plt.plot(masses[0:11],inner(rc_sly,ric_sly),color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,inner(rc_gm1,ric_gm1),color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("innerradius.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Outer Crust Thickness")
ax.set_xlabel(r"Mass $m(r)$ $[M\odot]$")
ax.set_ylabel(r"Outer crust thickness $[km]$")
plt.plot(masses[0:11],inner(ric_sly,r_sly),color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(masses,inner(ric_gm1,r_gm1),color='forestgreen',linestyle='dashed',label="GM1 EoS")
plt.legend(loc="best")
plt.savefig("outerradius.png")
plt.close()

plt.figure(figsize=(18,9))
ax=plt.gca()
ax.set_title(r"Density profiles")
ax.set_xlabel(r"Distance from the center $r$ $[km]$")
ax.set_ylabel(r"Density $\rho(r)$ $[g/cm_{3}]$")
nValues = np.array([0.0177,-0.0969,-0.0792,-0.0792,0.0631,-0.1438,0.0754,-0.0438,-0.0838,-0.1985,-0.0838,0.1146,-0.0385,-0.0446,-0.0985,-0.0192,0.0177])
datasets=[dd2rho,bskrho,apr3rho,engrho,tmarho,wff2rho,h4rho,mpa1rho,sfhoyrho,wff1rho,sfhorho,tm1rho,aprldrho,dd2frho,slyrho,alf2rho,bhblprho]
xValues=[dd2r,bskr,apr3r,engr,tmar,wff2r,h4r,mpa1r,sfhoyr,wff1r,sfhor,tm1r,aprldr,dd2fr,slyr,alf2r,bhblpr]
# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
colormap = cm.jet

# plot
for i in range(0,17,1):
    plt.plot(xValues[i],datasets[i], color=colormap(normalize(nValues[i])))

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(nValues)
plt.colorbar(scalarmappaple)

# show the figure
plt.show()
plt.savefig("densitiescolor.png")
plt.close()
Fpinsly=[]
x1=np.linspace(0,1171860,11720)
for i in range (0,1101628,100):
    z=np.sqrt(1132860**2-i**2)-np.sqrt(1101628**2-i**2)
    Fpinsly.append(2*z*10**15/2*1.2807)
for i in range (1101628,1132860,100):
    z=np.sqrt(1132860**2-i**2)
    Fpinsly.append(2*z*10**15/2*1.2807)
for i in range(1132860,1171860,100):
    Fpinsly.append(0)
plt.figure(figsize=(15,9))
Fpintm1=[]
x2=np.linspace(0,1448784,14489)
for i in range (0,1293645,100):
        z=np.sqrt(1448784**2-i**2)-np.sqrt(1293645**2-i**2)
        Fpintm1.append(2*z*10**15/2)
for i in range (1293645,1448784,100):
        z=np.sqrt(1448784**2-i**2)
        Fpintm1.append(2*z*10**15/2)
plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Pinning force $F_{pin}$")
ax.set_xlabel(r"x [cm]")
ax.set_ylabel(r"$F_{pin}$ [dyn]")
plt.plot(x1,Fpinsly,color='forestgreen',linestyle='dashed',label="SLy")
#plt.plot(x2,Fpintm1,color='orchid',linestyle='dashed',label="TM1")
plt.legend(loc="best")
print(Fpinsly[0])
print(max(Fpinsly))