import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import matplotlib.cm as cm

global G,c
G=6.67e-8
c=3e10

#Interpolating the EOS
columns = ["a", "b", "rho", "P"]

#Bool 0
bsk20=pd.read_csv("./eos/bsk20.csv", usecols=columns)[1::6]
rhobs=bsk20.rho
Pbs=bsk20.P
cPbs=CubicSpline(rhobs,Pbs)
crbs=CubicSpline(Pbs,rhobs)

#Bool 1
dd2=pd.read_csv("./eos/dd2.csv", usecols=columns)[1::4]
rhodd2=dd2.rho
Pdd2=dd2.P
cPdd2=CubicSpline(rhodd2,Pdd2)
crdd2=CubicSpline(Pdd2,rhodd2)

#Bool 2
apr=pd.read_csv("./eos/apr3.csv", usecols=columns)
rhoapr=apr.rho
Papr=apr.P
cPapr=CubicSpline(rhoapr,Papr)
crapr=CubicSpline(Papr,rhoapr)

#Bool 3
eng=pd.read_csv("./eos/eng.csv", usecols=columns)
rhoeng=eng.rho
Peng=eng.P
cPeng=CubicSpline(rhoeng,Peng)
creng=CubicSpline(Peng,rhoeng)

#Bool 4
tma=pd.read_csv("./eos/tma.csv", usecols=columns)[1::3]
rhotma=tma.rho
Ptma=tma.P
cPtma=CubicSpline(rhotma,Ptma)
crtma=CubicSpline(Ptma,rhotma)

#Bool 5
wff2=pd.read_csv("./eos/wff2.csv", usecols=columns)
rhowff2=wff2.rho
Pwff2=wff2.P
cPwff2=CubicSpline(rhowff2,Pwff2)
crwff2=CubicSpline(Pwff2,rhowff2)

#Bool 6
h4=pd.read_csv("./eos/h4.csv", usecols=columns)
rhoh4=h4.rho
Ph4=h4.P
cPh4=CubicSpline(rhoh4,Ph4)
crh4=CubicSpline(Ph4,rhoh4)

#Bool 7
mpa1=pd.read_csv("./eos/mpa1.csv", usecols=columns)[1::4]
rhompa1=mpa1.rho
Pmpa1=mpa1.P
cPmpa1=CubicSpline(rhompa1,Pmpa1)
crmpa1=CubicSpline(Pmpa1,rhompa1)

#Bool 8
sfhoy=pd.read_csv("./eos/sfhoy.csv", usecols=columns)[1::4]
rhosfhoy=sfhoy.rho
Psfhoy=sfhoy.P
cPsfhoy=CubicSpline(rhosfhoy,Psfhoy)
crsfhoy=CubicSpline(Psfhoy,rhosfhoy)

#Bool 9
wff1=pd.read_csv("./eos/wff1.csv", usecols=columns)
rhowff1=wff1.rho
Pwff1=wff1.P
cPwff1=CubicSpline(rhowff1,Pwff1)
crwff1=CubicSpline(Pwff1,rhowff1)

#Bool 10
sfho=pd.read_csv("./eos/sfho.csv", usecols=columns)[1::4]
rhosfho=sfho.rho
Psfho=sfho.P
cPsfho=CubicSpline(rhosfho,Psfho)
crsfho=CubicSpline(Psfho,rhosfho)

#Bool 12
tm1=pd.read_csv("./eos/tm1.csv", usecols=columns)[1::4]
rhotm1=tm1.rho
Ptm1=tm1.P
cPtm1=CubicSpline(rhotm1,Ptm1)
crtm1=CubicSpline(Ptm1,rhotm1)

#Bool 13
aprld=pd.read_csv("./eos/aprld.csv", usecols=columns)[1::4]
rhoaprld=aprld.rho
Paprld=aprld.P
cPaprld=CubicSpline(rhoaprld,Paprld)
craprld=CubicSpline(Paprld,rhoaprld)

#Bool 14
bhblp=pd.read_csv("./eos/bhblp.csv", usecols=columns)[1::4]
rhobhblp=bhblp.rho
Pbhblp=bhblp.P
cPbhblp=CubicSpline(rhobhblp,Pbhblp)
crbhblp=CubicSpline(Pbhblp,rhobhblp)

#Bool 15
dd2f=pd.read_csv("./eos/dd2f.csv", usecols=columns)
rhodd2f=dd2f.rho
Pdd2f=dd2f.P
cPdd2f=CubicSpline(rhodd2f,Pdd2f)
crdd2f=CubicSpline(Pdd2f,rhodd2f)

#Bool 17
sly=pd.read_csv("./eos/sly.csv", usecols=columns)
rhosly=sly.rho
Psly=sly.P
cPsly=CubicSpline(rhosly,Psly)
crsly=CubicSpline(Psly,rhosly)

#Bool 18
fps=pd.read_csv("./eos/fps.csv", usecols=columns)
rhofps=fps.rho
Pfps=fps.P
cPfps=CubicSpline(rhofps,Pfps)
crfps=CubicSpline(Pfps,rhofps)

#Bool 19
alf2=pd.read_csv("./eos/alf2.csv", usecols=columns)
rhoalf2=alf2.rho
Palf2=alf2.P
cPalf2=CubicSpline(rhoalf2,Palf2)
cralf2=CubicSpline(Palf2,rhoalf2)

#Returning the spatial derivatives of the functions
def f(x,bool):
    r=x[0]
    m=x[1]
    P=x[2]
    if(bool==0):
        rho=crdd2(P)
    elif(bool==1):
        rho=crbs(P)
    elif(bool==2):
        rho=crapr(P)
    elif(bool==3):
        rho=creng(P)
    elif(bool==4):
        rho=crtma(P)
    elif(bool==5):
        rho=crwff2(P)
    elif(bool==6):
        rho=crh4(P)
    elif(bool==7):
        rho=crmpa1(P)
    elif(bool==8):
        rho=crsfhoy(P)
    elif(bool==9):
        rho=crwff1(P)
    elif(bool==10):
        rho=crsfho(P)
    elif(bool==12):
        rho=crtm1(P)
    elif(bool==13):
        rho=craprld(P)
    elif(bool==14):
        rho=crbhblp(P)
    elif(bool==15):
        rho=crdd2f(P)
    elif(bool==17):
        rho=crsly(P)
    elif(bool==18):
        rho=crfps(P)
    elif(bool==19):
        rho=cralf2(P)
    dr_dr=1
    dm_dr=4.*np.pi*(r**2)*rho
    dP_dr=-(((G*m*rho)/(r**2))*(1+(P/(rho*c*c)))*(1+((4*np.pi*P*(r**3))/(m*c*c))))/(1-((2*G*m)/(r*c*c)))
    
    return np.array([dr_dr, dm_dr, dP_dr])

def ns_solve(rho_0,bool):
    #Initial Conditions
    dr=500 #In cm
    if(bool==0):
        P_0=cPdd2(rho_0)
    elif(bool==1):
        P_0=cPbs(rho_0)
    elif(bool==2):
        P_0=cPapr(rho_0)
    elif(bool==3):
        P_0=cPeng(rho_0)
    elif(bool==4):
        P_0=cPtma(rho_0)
    elif(bool==5):
        P_0=cPwff2(rho_0)
    elif(bool==6):
        P_0=cPh4(rho_0)
    elif(bool==7):
        P_0=cPmpa1(rho_0)
    elif(bool==8):
        P_0=cPsfhoy(rho_0)
    elif(bool==9):
        P_0=cPwff1(rho_0)
    elif(bool==10):
        P_0=cPsfho(rho_0)
    elif(bool==12):
        P_0=cPtm1(rho_0)
    elif(bool==13):
        P_0=cPaprld(rho_0)
    elif(bool==14):
        P_0=cPbhblp(rho_0)
    elif(bool==15):
        P_0=cPdd2f(rho_0)
    elif(bool==17):
        P_0=cPsly(rho_0)
    elif(bool==18):
        P_0=cPfps(rho_0)
    elif(bool==19):
        P_0=cPalf2(rho_0)
    X=np.zeros([3,80000])
    X[:,0]=np.array([500,1,P_0])

    #Solve using RK4
    for i in range(1,80000):
        k1=f(X[:,i-1],bool)
        k2=f(X[:,i-1]+k1*0.5*dr,bool)
        k3=f(X[:,i-1]+k2*0.5*dr,bool)
        k4=f(X[:,i-1]+k3*dr,bool)
        
        X[:,i]=X[:,i-1]+(dr*(k1+2*k2+2*k3+k4))/6.
        if((X[2,i]/P_0)<1e-10):
            break

    
    return X[:,i-1]

rho=np.arange(2.5e14,1e15,0.5e13)
rho=np.append(rho,np.arange(1e15,4e15,0.5e14))
res_dd2=np.zeros([3,len(rho)])
res_bsk=np.zeros([3,len(rho)])
res_apr=np.zeros([3,len(rho)])
res_eng=np.zeros([3,len(rho)])
res_tma=np.zeros([3,len(rho)])
res_wff2=np.zeros([3,len(rho)])
res_h4=np.zeros([3,len(rho)])
res_mpa1=np.zeros([3,len(rho)])
res_sfhoy=np.zeros([3,len(rho)])
res_wff1=np.zeros([3,len(rho)])
res_sfho=np.zeros([3,len(rho)])
res_tm1=np.zeros([3,len(rho)])
res_aprld=np.zeros([3,len(rho)])
res_bhblp=np.zeros([3,len(rho)])
res_dd2f=np.zeros([3,len(rho)])
res_sly=np.zeros([3,len(rho)])
res_fps=np.zeros([3,len(rho)])
res_alf2=np.zeros([3,len(rho)])

for i in range(len(rho)):
    res_dd2[:,i]=ns_solve(rho[i],0)
    res_bsk[:,i]=ns_solve(rho[i],1)
    res_apr[:,i]=ns_solve(rho[i],2)
    res_eng[:,i]=ns_solve(rho[i],3)
    res_tma[:,i]=ns_solve(rho[i],4)
    res_wff2[:,i]=ns_solve(rho[i],5)
    res_h4[:,i]=ns_solve(rho[i],6)
    res_mpa1[:,i]=ns_solve(rho[i],7)
    res_sfhoy[:,i]=ns_solve(rho[i],8)
    res_wff1[:,i]=ns_solve(rho[i],9)
    res_sfho[:,i]=ns_solve(rho[i],10)
    res_tm1[:,i]=ns_solve(rho[i],12)
    res_aprld[:,i]=ns_solve(rho[i],13)
    res_bhblp[:,i]=ns_solve(rho[i],14)
    res_dd2f[:,i]=ns_solve(rho[i],15)
    res_sly[:,i]=ns_solve(rho[i],17)
    res_fps[:,i]=ns_solve(rho[i],18)
    res_alf2[:,i]=ns_solve(rho[i],19)
    print(i)

R_dd2=res_dd2[0,]/1e5
R_bsk=res_bsk[0,]/1e5
R_apr=res_apr[0,]/1e5
R_eng=res_eng[0,]/1e5
R_tma=res_tma[0,]/1e5
R_wff2=res_wff2[0,]/1e5
R_h4=res_h4[0,]/1e5
R_mpa1=res_mpa1[0,]/1e5
R_sfhoy=res_sfhoy[0,]/1e5
R_wff1=res_wff1[0,]/1e5
R_sfho=res_sfho[0,]/1e5
R_tm1=res_tm1[0,]/1e5
R_aprld=res_aprld[0,]/1e5
R_bhblp=res_bhblp[0,]/1e5
R_dd2f=res_dd2f[0,]/1e5
R_sly=res_sly[0,]/1e5
R_fps=res_fps[0,]/1e5
R_alf2=res_alf2[0,]/1e5
M_dd2=res_dd2[1,]/2e33
M_bsk=res_bsk[1,]/2e33
M_apr=res_apr[1,]/2e33
M_eng=res_eng[1,]/2e33
M_tma=res_tma[1,]/2e33
M_wff2=res_wff2[1,]/2e33
M_h4=res_h4[1,]/2e33
M_mpa1=res_mpa1[1,]/2e33
M_sfhoy=res_sfhoy[1,]/2e33
M_wff1=res_wff1[1,]/2e33
M_sfho=res_sfho[1,]/2e33
M_tm1=res_tm1[1,]/2e33
M_aprld=res_aprld[1,]/2e33
M_bhblp=res_bhblp[1,]/2e33
M_dd2f=res_dd2f[1,]/2e33
M_sly=res_sly[1,]/2e33
M_fps=res_fps[1,]/2e33
M_alf2=res_alf2[1,]/2e33
#List of maximum mass and its radius for every EOS
eoslist=["DD2","BSk20","APR3","ENG","TMA","WFF2","H4","MPA1","SFHOY","WFF1","SFHO","TM1","APRLD","BHBLP","DD2F","SLy","FPS","ALF2"]
maxmass=[max(M_dd2),max(M_bsk),max(M_apr),max(M_eng),max(M_tma),max(M_wff2),max(M_h4),max(M_mpa1),max(M_sfhoy),max(M_wff1),max(M_sfho),max(M_tm1),max(M_aprld),max(M_bhblp),max(dd2f),max(M_sly),max(M_fps),max(M_alf2)]
maxradius=[R_dd2[np.argmax(M_dd2)],R_bsk[np.argmax(M_bsk)],R_apr[np.argmax(M_apr)],R_eng[np.argmax(M_eng)],R_tma[np.argmax(M_tma)],R_wff2[np.argmax(M_wff2)],R_h4[np.argmax(M_h4)],R_mpa1[np.argmax(M_mpa1)],R_sfhoy[np.argmax(M_sfhoy)],R_wff1[np.argmax(M_wff1)],R_sfho[np.argmax(M_sfho)],R_tm1[np.argmax(M_tm1)],R_aprld[np.argmax(M_aprld)],R_bhblp[np.argmax(M_bhblp)],R_dd2f[np.argmax(M_dd2f)],R_sly[np.argmax(M_sly)],R_fps[np.argmax(M_fps)],R_alf2[np.argmax(M_alf2)]]
with open("maxmass.txt", "w") as txt_file:
    txt_file.write("EOS,"+"Max Mass (M_sun),"+ "Radius (km)" + "\n")
    for i in range (0,18,1):
        txt_file.write(eoslist[i]+" "+str(maxmass[i])+" "+str(maxradius[i])+ "\n") # works with any number of elements in a line

#.txt files with M(R) for every EOS
with open("dd2.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_dd2)-1,1):
        txt_file.write(str(M_dd2[i])+" "+str(R_dd2[i]) + "\n")
with open("bsk20.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_bsk)-1,1):
        txt_file.write(str(M_bsk[i])+" "+str(R_bsk[i]) + "\n")
with open("apr3.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_apr)-1,1):
        txt_file.write(str(M_apr[i])+" "+str(R_apr[i]) + "\n")
with open("eng.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_eng)-1,1):
        txt_file.write(str(M_eng[i])+" "+str(R_eng[i]) + "\n")
with open("tma.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_tma)-1,1):
        txt_file.write(str(M_tma[i])+" "+str(R_tma[i]) + "\n")
with open("wff2.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_wff2)-1,1):
        txt_file.write(str(M_wff2[i])+" "+str(R_wff2[i]) + "\n")
with open("h4.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_h4)-1,1):
        txt_file.write(str(M_h4[i])+" "+str(R_h4[i]) + "\n")
with open("mpa1.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_mpa1)-1,1):
        txt_file.write(str(M_mpa1[i])+" "+str(R_mpa1[i]) + "\n")
with open("sfhoy.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_sfhoy)-1,1):
        txt_file.write(str(M_sfhoy[i])+" "+str(R_sfhoy[i]) + "\n")
with open("wff1.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_wff1)-1,1):
        txt_file.write(str(M_wff1[i])+" "+str(R_wff1[i]) + "\n")
with open("sfho.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_sfho)-1,1):
        txt_file.write(str(M_sfho[i])+" "+str(R_sfho[i]) + "\n")
with open("tm1.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_tm1)-1,1):
        txt_file.write(str(M_tm1[i])+" "+str(R_tm1[i]) + "\n")
with open("aprld.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_aprld)-1,1):
        txt_file.write(str(M_aprld[i])+" "+str(R_aprld[i]) + "\n")
with open("bhblp.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_bhblp)-1,1):
        txt_file.write(str(M_bhblp[i])+" "+str(R_bhblp[i]) + "\n")
with open("dd2f.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_dd2f)-1,1):
        txt_file.write(str(M_dd2f[i])+" "+str(R_dd2f[i]) + "\n")
with open("sly.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_sly)-1,1):
        txt_file.write(str(M_sly[i])+" "+str(R_sly[i]) + "\n")
with open("fps.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_fps)-1,1):
        txt_file.write(str(M_fps[i])+" "+str(R_fps[i]) + "\n")
with open("alf2.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Radius (km)" + "\n")
    for i in range (0,len(M_alf2)-1,1):
        txt_file.write(str(M_alf2[i])+" "+str(R_alf2[i]) + "\n")

        

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Mass vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"Mass of the Star [$M_\odot$]")
plt.plot(rho,M_dd2,color='black',label="dd2 EoS")
plt.plot(rho,M_bsk,color='forestgreen',label="BSk20 EoS")
plt.plot(rho,M_apr,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(rho,M_eng,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(rho,M_tma,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(rho,M_wff2,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(rho,M_h4,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(rho,M_mpa1,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(rho,M_sfhoy,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(rho,M_wff1,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(rho,M_sfho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(rho,M_tm1,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(rho,M_aprld,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(rho,M_bhblp,color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(rho,M_dd2f,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(rho,M_sly,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(rho,M_fps,color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(rho,M_alf2,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("MvsrhoStat.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: $\rho_c$ vs Mass")
ax.set_ylabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_xlabel(r"Mass of the Star [$M_\odot$]")
plt.plot(M_dd2,rho,color='black',label="dd2 EoS")
plt.plot(M_bsk,rho,color='forestgreen',label="BSk20 EoS")
plt.plot(M_apr,rho,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(M_eng,rho,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(M_tma,rho,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(M_wff2,rho,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(M_h4,rho,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(M_mpa1,rho,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(M_sfhoy,rho,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(M_wff1,rho,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(M_sfho,rho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(M_tm1,rho,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(M_aprld,rho,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(M_bhblp,rho,color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(M_dd2f,rho,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(M_sly,rho,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(M_fps,rho,color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(M_alf2,rho,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("rhovsMStat.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Radius vs $\rho_c$")
ax.set_xlabel(r"$\rho_c$ [$g/cm^3$]")
ax.set_ylabel(r"Radius of the Star [km]")
ax.set_ylim([9, 30])
plt.plot(rho,R_dd2,color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(rho,R_bsk,color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(rho,R_apr,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(rho,R_eng,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(rho,R_tma,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(rho,R_wff2,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(rho,R_h4,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(rho,R_mpa1,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(rho,R_sfhoy,color='slategray', linestyle='dashed',label="SFHOY EoS")
plt.plot(rho,R_wff1,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(rho,R_sfho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(rho,R_tm1,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(rho,R_aprld,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(rho,R_bhblp,color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(rho,R_dd2f,color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(rho,R_sly,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(rho,R_fps,color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(rho,R_alf2,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("RvsrhoStat.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Radius vs Mass")
ax.set_xlabel(r"Radius of the Star [km]")
ax.set_ylabel(r"Mass of the Star [$M_\odot$]")
ax.set_xlim([8, 20])
ax.set_ylim([0,3])
plt.plot(R_dd2,M_dd2, color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(R_bsk,M_bsk,color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(R_apr,M_apr,color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(R_eng,M_eng,color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(R_tma,M_tma,color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(R_wff2,M_wff2,color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(R_h4,M_h4,color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(R_mpa1,M_mpa1,color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(R_sfhoy,M_sfhoy,color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(R_wff1,M_wff1,color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(R_sfho,M_sfho,color='darkslategray',linestyle='dashed',label="SFHO EoS")
#plt.plot(R_eosAU,M_eosAU,color='darksalmon',linestyle='dashed',label="eosAU EoS")
plt.plot(R_tm1,M_tm1,color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(R_aprld,M_aprld,color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(R_bhblp,M_bhblp,color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(R_dd2f,M_dd2f,color='violet',linestyle='dashed',label="DD2F EoS")
#plt.plot(R_eosUU,M_eosUU,color='dodgerblue',linestyle='dashed',label="eosUU EoS")
plt.plot(R_sly,M_sly,color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(R_fps,M_fps,color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(R_alf2,M_alf2,color='lawngreen',linestyle='dashed',label="ALF2 EoS")
r = np.linspace(0,15,100)
m1 = 1/2 * r / 1.474 #black hole limit
m2 = 4/9 * r / 1.474 #Buchdahl limit
m3 = 0.354 * r / 1.474 #causality limit
plt.plot(r, m1, color = 'black', lw=1.5) #Black hole
plt.fill_between(r, m1, 3, color = 'black', label='Black Hole')
plt.text(4, 2, 'Black hole', color = 'white', rotation = 45)
plt.plot(r, m2, color = 'dimgrey', lw=1.5) #Buchdahl
plt.fill_between(r, m2, m1, color = 'dimgrey',label='Budhahl')
plt.text(5, 1.6, 'Buchdahl limit', color = 'white', rotation = 42)
plt.plot(r, m3, color = 'darkgrey', lw=1.5) #Causality
plt.fill_between(r, m3, m2, color = 'darkgray', label='Causality')
plt.text(5.5, 1.42, 'Causality limit', color = 'white', rotation = 36)
plt.legend(loc="best")
plt.savefig("RvsMStat.png")
plt.close()

plt.figure(figsize=(18,9))
ax.set_title(r"Stationary NS Plot: Radius vs Mass")
ax.set_xlabel(r"Radius of the Star [km]")
ax.set_ylabel(r"Mass of the Star [$M_\odot$]")
ax.set_xlim([8, 20])
ax.set_ylim([0,3])
nValues = np.array([0.0177,-0.0969,-0.0792,-0.0792,0.0631,-0.1438,0.0754,-0.0438,-0.0838,-0.1985,-0.0838,0.1146,-0.0385,-0.0446,-0.0985,-0.0192,0.0177,-0.1681])
xValues=[R_dd2,R_bsk,R_apr,R_eng,R_tma,R_wff2,R_h4,R_mpa1,R_sfhoy,R_wff1,R_sfho,R_tm1,R_aprld,R_dd2f,R_sly,R_alf2,R_bhblp,R_fps]
datasets=[M_dd2,M_bsk,M_apr,M_eng,M_tma,M_wff2,M_h4,M_mpa1,M_sfhoy,M_wff1,M_sfho,M_tm1,M_aprld,M_dd2f,M_sly,M_alf2,M_bhblp,M_fps]
# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
colormap = cm.jet

# plot
for i in range(0,18,1):
    plt.plot(xValues[i],datasets[i], color=colormap(normalize(nValues[i])))

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(nValues)
plt.colorbar(scalarmappaple)

# show the figure
plt.show()
plt.savefig("RvsMStatcolor.png")
plt.close()

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Stationary NS Plot: Radius vs Mass 2")
ax.set_xlabel(r"Radius of the Star [km]")
ax.set_ylabel(r"Mass of the Star [$M_\odot$]")
ax.set_xlim([8, 20])
plt.plot(R_dd2[:np.argmax(M_dd2)],M_dd2[:np.argmax(M_dd2)], color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(R_bsk[:np.argmax(M_bsk)],M_bsk[:np.argmax(M_bsk)],color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(R_apr[:np.argmax(M_apr)],M_apr[:np.argmax(M_apr)],color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(R_eng[:np.argmax(M_eng)],M_eng[:np.argmax(M_eng)],color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(R_tma[:np.argmax(M_tma)],M_tma[:np.argmax(M_tma)],color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(R_wff2[:np.argmax(M_wff2)],M_wff2[:np.argmax(M_wff2)],color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(R_h4[:np.argmax(M_h4)],M_h4[:np.argmax(M_h4)],color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(R_mpa1[:np.argmax(M_mpa1)],M_mpa1[:np.argmax(M_mpa1)],color='peru', linestyle='dashed',label="MPA1 EoS")
plt.plot(R_sfhoy[:np.argmax(M_sfhoy)],M_sfhoy[:np.argmax(M_sfhoy)],color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(R_wff1[:np.argmax(M_wff1)],M_wff1[:np.argmax(M_wff1)],color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(R_sfho[:np.argmax(M_sfho)],M_sfho[:np.argmax(M_sfho)],color='darkslategray',linestyle='dashed',label="SFHO EoS")
#plt.plot(R_eosAU[:np.argmax(M_eosAU)],M_eosAU[:np.argmax(M_eosAU)],color='darksalmon',linestyle='dashed',label="eosAU EoS")
plt.plot(R_tm1[:np.argmax(M_tm1)],M_tm1[:np.argmax(M_tm1)],color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(R_aprld[:np.argmax(M_aprld)],M_aprld[:np.argmax(M_aprld)],color='thistle',linestyle='dashed',label="APRLD EoS")
plt.plot(R_bhblp[:np.argmax(M_bhblp)],M_bhblp[:np.argmax(M_bhblp)],color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(R_dd2f[:np.argmax(M_dd2f)],M_dd2f[:np.argmax(M_dd2f)],color='violet',linestyle='dashed',label="DD2F EoS")
#plt.plot(R_eosUU,M_eosUU,color='dodgerblue',linestyle='dashed',label="eosUU EoS")
plt.plot(R_sly[:np.argmax(M_sly)],M_sly[:np.argmax(M_sly)],color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(R_fps[:np.argmax(M_fps)],M_fps[:np.argmax(M_fps)],color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(R_alf2[:np.argmax(M_alf2)],M_alf2[:np.argmax(M_alf2)],color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("RvsMStat2.png")
plt.close()

with open("slyrho.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Density" + "\n")
    for i in range (0,len(M_sly)-1,1):
        txt_file.write(str(M_sly[i])+" "+str(rho[i]) + "\n")

#inertia
mdd2=M_dd2[:np.argmax(M_dd2)]
mbsk=M_bsk[:np.argmax(M_bsk)]
mapr=M_apr[:np.argmax(M_apr)]
meng=M_eng[:np.argmax(M_eng)]
mtma=M_tma[:np.argmax(M_tma)]
mwff2=M_wff2[:np.argmax(M_wff2)]
mh4=M_h4[:np.argmax(M_h4)]
mmpa1=M_mpa1[:np.argmax(M_mpa1)]
msfhoy=M_sfhoy[:np.argmax(M_sfhoy)]
mwff1=M_wff1[:np.argmax(M_wff1)]
msfho=M_sfho[:np.argmax(M_sfho)]
mtm1=M_tm1[:np.argmax(M_tm1)]
maprld=M_aprld[:np.argmax(M_aprld)]
mbhblp=M_bhblp[:np.argmax(M_bhblp)]
mdd2f=M_dd2f[:np.argmax(M_dd2f)]
msly=M_sly[:np.argmax(M_sly)]
mfps=M_fps[:np.argmax(M_fps)]
malf2=M_alf2[:np.argmax(M_alf2)]
rdd2=R_dd2[:np.argmax(M_dd2)]
rbsk=R_bsk[:np.argmax(M_bsk)]
rapr=R_apr[:np.argmax(M_apr)]
reng=R_eng[:np.argmax(M_eng)]
rtma=R_tma[:np.argmax(M_tma)]
rwff2=R_wff2[:np.argmax(M_wff2)]
rh4=R_h4[:np.argmax(M_h4)]
rmpa1=R_mpa1[:np.argmax(M_mpa1)]
rsfhoy=R_sfhoy[:np.argmax(M_sfhoy)]
rwff1=R_wff1[:np.argmax(M_wff1)]
rsfho=R_sfho[:np.argmax(M_sfho)]
rtm1=R_tm1[:np.argmax(M_tm1)]
raprld=R_aprld[:np.argmax(M_aprld)]
rbhblp=R_bhblp[:np.argmax(M_bhblp)]
rdd2f=R_dd2f[:np.argmax(M_dd2f)]
rsly=R_sly[:np.argmax(M_sly)]
rfps=R_fps[:np.argmax(M_fps)]
ralf2=R_alf2[:np.argmax(M_alf2)]

def inertia(m,r):
    return (0.237+0.008)*m*r**2*1.989*10**43*(1+(4.2*(m/r))+90*(m/r)**4)

plt.figure(figsize=(15,9))
ax=plt.gca()
ax.set_title(r"Moment of Inertia")
ax.set_xlabel(r"Mass [$M\odot$]")
ax.set_ylabel(r"Moment of Inertia [$10^{45} g*cm^{2}$]")
ax.set_xlim([1, 2.7])
plt.plot(mdd2,inertia(mdd2,rdd2),color='black',linestyle='dashed',label="dd2 EoS")
plt.plot(mbsk,inertia(mbsk,rbsk),color='forestgreen',linestyle='dashed',label="BSk20 EoS")
plt.plot(mapr,inertia(mapr,rapr),color='purple',linestyle='dashed',label="APR3 EoS")
plt.plot(meng,inertia(meng,reng),color='navy',linestyle='dashed',label="ENG EoS")
plt.plot(mtma,inertia(mtma,rtma),color='firebrick',linestyle='dashed',label="TMA EoS")
plt.plot(mwff2,inertia(mwff2,rwff2),color='cyan',linestyle='dashed',label="WFF2 EoS")
plt.plot(mh4,inertia(mh4,rh4),color='hotpink',linestyle='dashed',label="H4 EoS")
plt.plot(mmpa1,inertia(mmpa1,rmpa1),color='peru',linestyle='dashed',label="MPA1 EoS")
plt.plot(msfhoy,inertia(msfhoy,rsfhoy),color='slategray',linestyle='dashed',label="SFHOY EoS")
plt.plot(mwff1,inertia(mwff1,rwff1),color='darkseagreen',linestyle='dashed',label="WFF1 EoS")
plt.plot(msfho,inertia(msfho,rsfho),color='darkslategray',linestyle='dashed',label="SFHO EoS")
plt.plot(mtm1,inertia(mtm1,rtm1),color='moccasin',linestyle='dashed',label="TM1 EoS")
plt.plot(maprld,inertia(maprld,raprld),color='thistle',linestyle='dashed',label="APMLD EoS")
plt.plot(mbhblp,inertia(mbhblp,rbhblp),color='steelblue',linestyle='dashed',label="BHBLD EoS")
plt.plot(mdd2f,inertia(mdd2f,rdd2f),color='violet',linestyle='dashed',label="DD2F EoS")
plt.plot(msly,inertia(msly,rsly),color='indianred',linestyle='dashed',label="SLy EoS")
plt.plot(mfps,inertia(mfps,rfps),color='olive',linestyle='dashed',label="FPS EoS")
plt.plot(malf2,inertia(malf2,ralf2),color='lawngreen',linestyle='dashed',label="ALF2 EoS")
plt.legend(loc="best")
plt.savefig("inertia.png")
plt.close()

with open("slyin.txt", "w") as txt_file:
    txt_file.write("Mass (M_sun) "+ "Inertia" + "\n")
    for i in range (0,len(msly)-1,1):
        txt_file.write(str(msly[i])+" "+str(inertia(msly[i],rsly[i])) + "\n")

plt.figure(figsize=(15,9))
ax.set_title(r"Moment of Inertia")
ax.set_xlabel(r"Mass of the Star [$M_{\odot}$]")
ax.set_ylabel(r"Moment of Inertia [$10^{45} g*cm^{2}$]")
ax.set_ylim([1,2.7])
nValues = np.array([0.0177,-0.0969,-0.0792,-0.0792,0.0631,-0.1438,0.0754,-0.0438,-0.0838,-0.1985,-0.0838,0.1146,-0.0385,-0.0446,-0.0985,-0.0192,0.0177,-0.1681])
xValues=[mdd2,mbsk,mapr,meng,mtma,mwff2,mh4,mmpa1,msfhoy,mwff1,msfho,mtm1,maprld,mdd2f,msly,malf2,mbhblp,mfps]
datasets=[inertia(mdd2,rdd2),inertia(mbsk,rbsk),inertia(mapr,rapr),inertia(meng,reng),inertia(mtma,rtma),inertia(mwff2,rwff2),inertia(mh4,rh4),inertia(mmpa1,rmpa1),inertia(msfhoy,rsfhoy),inertia(mwff1,rwff1),inertia(msfho,rsfho),inertia(mtm1,rtm1),inertia(maprld,raprld),inertia(mdd2f,rdd2f),inertia(msly,rsly),inertia(mbhblp,rbhblp),inertia(malf2,ralf2),inertia(mfps,rfps)]
# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
colormap = cm.jet

# plot
for i in range(0,18,1):
    plt.plot(xValues[i],datasets[i], color=colormap(normalize(nValues[i])))

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(nValues)
plt.colorbar(scalarmappaple)

# show the figure
plt.show()
plt.savefig("inertiacolor.png")
plt.close()