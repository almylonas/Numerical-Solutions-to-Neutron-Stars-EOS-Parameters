import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd

columns = ["a", "b", "rho", "P"]

bsk20=pd.read_csv("./eos/bsk20.csv", usecols=columns)[1::6]
rhobs=bsk20.rho
Pbs=bsk20.P
cPbs=CubicSpline(rhobs,Pbs)

dd2=pd.read_csv("./eos/dd2.csv", usecols=columns)[1::4]
rhodd2=dd2.rho
Pdd2=dd2.P
cPdd2=CubicSpline(rhodd2,Pdd2)

apr=pd.read_csv("./eos/apr3.csv", usecols=columns)
rhoapr=apr.rho
Papr=apr.P
cPapr=CubicSpline(rhoapr,Papr)

eng=pd.read_csv("./eos/eng.csv", usecols=columns)
rhoeng=eng.rho
Peng=eng.P
cPeng=CubicSpline(rhoeng,Peng)

tma=pd.read_csv("./eos/tma.csv", usecols=columns)[1::3]
rhotma=tma.rho
Ptma=tma.P
cPtma=CubicSpline(rhotma,Ptma)

wff2=pd.read_csv("./eos/wff2.csv", usecols=columns)
rhowff2=wff2.rho
Pwff2=wff2.P
cPwff2=CubicSpline(rhowff2,Pwff2)

h4=pd.read_csv("./eos/h4.csv", usecols=columns)
rhoh4=h4.rho
Ph4=h4.P
cPh4=CubicSpline(rhoh4,Ph4)

mpa1=pd.read_csv("./eos/mpa1.csv", usecols=columns)[1::4]
rhompa1=mpa1.rho
Pmpa1=mpa1.P
cPmpa1=CubicSpline(rhompa1,Pmpa1)

sfhoy=pd.read_csv("./eos/sfhoy.csv", usecols=columns)[1::4]
rhosfhoy=sfhoy.rho
Psfhoy=sfhoy.P
cPsfhoy=CubicSpline(rhosfhoy,Psfhoy)

wff1=pd.read_csv("./eos/wff1.csv", usecols=columns)
rhowff1=wff1.rho
Pwff1=wff1.P

sfho=pd.read_csv("./eos/sfho.csv", usecols=columns)
rhosfho=sfho.rho
Psfho=sfho.P

tm1=pd.read_csv("./eos/tm1.csv", usecols=columns)[1::4]
rhotm1=tm1.rho
Ptm1=tm1.P

aprld=pd.read_csv("./eos/aprld.csv", usecols=columns)[1::4]
rhoaprld=aprld.rho
Paprld=aprld.P

bhblp=pd.read_csv("./eos/bhblp.csv", usecols=columns)[1::4]
rhobhblp=bhblp.rho
Pbhblp=bhblp.P

dd2f=pd.read_csv("./eos/dd2f.csv", usecols=columns)
rhodd2f=dd2f.rho
Pdd2f=dd2f.P

sly=pd.read_csv("./eos/sly.csv", usecols=columns)
rhosly=sly.rho
Psly=sly.P

fps=pd.read_csv("./eos/fps.csv", usecols=columns)
rhofps=fps.rho
Pfps=fps.P

alf2=pd.read_csv("./eos/alf2.csv", usecols=columns)
rhoalf2=alf2.rho
Palf2=alf2.P

fig=plt.figure(figsize=(15,9))
ax=plt.gca()
ax=plt.gca()
ax.set_title(r"Equations of State [$P(\rho)$]")
ax.set_xlabel(r"$\rho\: (g/cm^3)$")
ax.set_ylabel(r"$P\: (dyn/cm^2)$")
plt.xscale("log")
plt.yscale("log")
plt.plot(rhoeng,Peng,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(rhobs,Pbs,color='forestgreen',linestyle='dashed',label="BSk EoS")
plt.plot(rhodd2,Pdd2,color='black',linestyle='dashed',label="DD2 EoS")
plt.plot(rhotma,Ptma,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(rhowff2,Pwff2,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(rhoh4,Ph4,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(rhompa1,Pmpa1,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(rhosfhoy,Psfhoy,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(rhowff1,Pwff1,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(rhosfho,Psfho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(rhotm1,Ptm1,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(rhodd2f,Pdd2f,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(rhoapr,Papr,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(rhoaprld,Paprld,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(rhobhblp,Pbhblp,color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(rhosly,Psly,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(rhofps,Pfps,color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(rhoalf2,Palf2,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("eos.png")
plt.close()




