#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
Zmax = 32
abunds_idents={"h":1,"he":2,"li":3,"be":4,"b":5,"c":6,"n":7,"o":8,"f":9,"ne":10,"na":11,"mg":12,"al":13,"si":14,"p":15,"s":16,"cl":17,"ar":18,"k":19,"ca":20,"sc":21,"ti":22,"v":23,"cr":24,"mn":25,"fe":26,"co":27,"ni":28,"cu":29,"zn":30,"ga":31,"ge":32}

def abunds_dic(data):
    list_of_abunds=[]
    abunds={}
    abunds_data=np.hstack((data[:,4:],np.zeros((data.shape[0],3))))#33
    abunds_data=abunds_data[::-1,:]
    rows,columns=abunds_data.shape
    for j in xrange(rows):
        for k,i in abunds_idents.items():
            abunds[k]=abunds_data[j,i-1]#32
        list_of_abunds.append(abunds)
        abunds={}
    return list_of_abunds,rows

#criar uma lista de dicionarios (quantos forem o numero de linhas)
#fazer uma funcao, chamala quantas vezes forem o numero de linhas

def one_row_abundances(X6,fname):
    with open(fname,"w") as table:
        ind='#Index Z=1  -  Z=%d' % Zmax
        table.write("{0:>3s}\n".format(ind))
        #n=len(Xinitial)
        X6shell=np.zeros(Zmax+1)

        for k,i in X6.items():
            z=abunds_idents[k]
            X6shell[z]=i

        #duplicate line
        X6shelldup=X6shell.copy()

            #write index
        X6shelldup[0] = "1"

        #write data to file
        table.write("%s\n" % (" ".join(map(str,X6shell))))
        table.write("%s\n" % (" ".join(map(str,X6shelldup))))


def write_first_abundance_line(Xi,X6,fname):

    with open(fname,"w") as table:
        ind='#Index Z=1  -  Z=%d' % Zmax
        table.write("{0:>3s}\n".format(ind))
        #n=len(Xinitial)
        Xinitial=np.zeros(Zmax+1)
        X6shell=np.zeros(Zmax+1)

        for k, i in Xi.items():
            z = abunds_idents[k]
            Xinitial[z] = i

        for k,i in X6.items():
            z=abunds_idents[k]
            X6shell[z]=i

        #duplicate line
        X6shelldup=X6shell.copy()

            #write index
        X6shelldup[0] = "1"
        Xinitial[0] = "2"

        #write data to file
        table.write("%s\n" % (" ".join(map(str,X6shell))))
        table.write("%s\n" % (" ".join(map(str,X6shelldup))))
        table.write("%s\n" % (" ".join(map(str,Xinitial))))

def add_abundance_line(X, fname):
    with open(fname, "r") as table:
        #read the data
        data=np.loadtxt(fname,skiprows=1)
    #reshuffle the data, i.e. line shifts
    columns,rows=data.shape
    for i in xrange(columns):
        data[-i-1,0]=columns-i
    #data[-1,0]=2
    #data[-2,0]=1
    #data[-3,0]=1
    Xinitial=np.zeros(Zmax+1)
    #add new line, i.e. X
    for k,i in X.items():
        z=abunds_idents[k]
        Xinitial[z]=i

    #duplicate line
    Xinitial[0] = '1'
    data[-columns,0:]=Xinitial
    Xduplicate=Xinitial.copy()
    Xduplicate[0] = '0'
    new_data=np.append(np.array([Xduplicate]),data, axis=0)
    with open(fname, "w") as table:
        ind='#Index Z=1  -  Z=%d' % Zmax
        #print new_data.shape
        new_columns, new_rows = new_data.shape
        table.write("{0:>3s}\n".format(ind))
        for i in xrange(new_columns):
            table.write(" %s\n" % (" ".join(map(str,new_data[i,:]))))

#Write file with 2 shells
#for outershell and -6d shell 
#write_first_abundance_line({"c": 0.472155, "o": 0.472155,"na":0.00025,"mg":0.003,"al":0.00032,"si":0.005,"s":0.002,"ca":0.00003,"ti":0.00004,"cr":0.00004,"fe":0.00001},{"c": 0.05, "o": 0.9363,"na":0.0025,"mg":0.0,"al":0.0032,"si":0.01,"s":0.003,"ca":0.001,"ti":0.0004,"cr":0.0004,"fe":0.0001})
#for -5d shell
#add_abundance_line({"o": 0.17,"na":0.0013,"mg":0.04,"al":0.0032,"si":0.65,"s":0.10,"ca":0.0003,"ti":0.01,"cr":0.01,"fe":0.0175},"AbundanceTable.dat")
#for -3d shell
#add_abundance_line({"o": 0.01,"al":0.0011,"si":0.69,"s":0.12,"ca":0.0003,"ti":0.05,"cr":0.05,"fe":0.08},"AbundanceTable.dat")
#for +4.8d shell
#add_abundance_line({"si":0.69,"s":0.06,"ca":0.0003,"ti":0.0675,"cr":0.0675,"fe":0.1011,"co":0.008015,"ni":0.0008397},"AbundanceTable.dat")
#for 12.d shell
#add_abundance_line({"si":0.64,"ca":0.0003,"ti":0.0350,"cr":0.0350,"fe":0.127837,"co":0.150807,"ni":0.0063555},"AbundanceTable.dat")

def write_abunds_table(data, _runid, nepoch):
    fname='Abundances_%.1f_%05d_%d.dat' %(data[0,2],_runid,nepoch)
    list_of_abunds,rows=abunds_dic(data)
    if rows == 1:
        one_row_abundances(list_of_abunds[0],fname)
    elif rows == 2:
        write_first_abundance_line(list_of_abunds[0],list_of_abunds[1],fname)
    else:
        write_first_abundance_line(list_of_abunds[0],list_of_abunds[1],fname)
        for i in xrange(2,rows):
            add_abundance_line(list_of_abunds[i],fname)
    return fname

def read_w7_hydro():

    return np.loadtxt("w7.hydro",skiprows=2)

def extrapolate(vmax,npoints):
    data=read_w7_hydro()
    radii=data[:,2]
    densities=data[:,3]
    w7_velocities=radii/(20*1e5)
    w7l=np.log10(w7_velocities)
    denl=np.log10(densities)
    xi=np.array([w7l[175],w7l[176]])
    yi=np.array([denl[175],denl[176]])
    v=np.linspace(22800,vmax,npoints)
    vl=np.log10(v)
    x=vl
    s=IUS(xi,yi,k=1)
    y=s(x)

    xr=10**x
    yr=10**y

    return xr,yr

def remap_density_by_analytical_integration(vmin,vmax,nshells,npoints):
    data=read_w7_hydro()
    #actual w7 velocities (don't follow homologous expansion)
    #vel=data[:,5]/1e5
    radii=data[:,2]
    densities=data[:,3]
    vs=np.linspace(1./vmin,1./vmax,nshells+1)
    N=len(vs)
    vbounds=[]
    for i in xrange(N):
        vb=1./vs[i]
        vbounds.append(vb)
    vdummy=vbounds[0]/2
    vbounds.insert(0,vdummy)
    w7_velocities=radii/(20*1e5)
    w7_vel=np.append(0,w7_velocities)
    if vmax>22800:
        ext_vel1,ext_den1=extrapolate(1.1 * vmax,npoints)
        ext_vel=ext_vel1[1:]
        ext_den=ext_den1[1:]
        w7_vel=np.append([w7_vel],[ext_vel])
        densities=np.append([densities],[ext_den])
    N=len(vbounds)
    WN=len(w7_vel)
    rho_w7=[]
    rho_tot=[]
    rho_total=[]

#    np.savetxt("extrap.dat", np.array([w7_vel[1:], densities]).T)

    for j in xrange(WN):
        if j==WN-1:
            break
        for i in xrange(N):
            if vbounds[i]<w7_vel[j+1]:
                if i==N-1:
                    #last hachinger cell
                    break
                        #partially in shell
                if vbounds[i+1]>w7_vel[j+1]:
                    rho = (
                        (densities[j] * (w7_vel[j + 1] ** 3 - w7_vel[j] ** 3))
                        / (vbounds[i + 1] ** 3 - vbounds[i] ** 3)
                        if vbounds[i] < w7_vel[j]
                        else (
                            densities[j]
                            * (w7_vel[j + 1] ** 3 - vbounds[i] ** 3)
                        )
                        / (vbounds[i + 1] ** 3 - vbounds[i] ** 3)
                    )
                    #print densities[j]
                    rho_w7.append(rho)
                else:
                    #completely in shell
                    rho=(densities[j]*(vbounds[i+1]**3-w7_vel[j]**3))/(vbounds[i+1]**3-vbounds[i]**3)
                    if rho>0:
                        rho_w7.append(rho)
                        #print densities[j]
                        rho_w7=np.array(rho_w7)
                        rho_t=rho_w7.sum()
                        rho_total.append(rho_t)
                        #print rho_total
                        rho_w7=[]
                        #print rho_w7
    return vbounds,rho_total
#ext_vel1,ext_den

def homologous_expansion(t0,t,vmin,vmax,nshells,npoints):
    vbounds,rho=remap_density_by_analytical_integration(vmin,vmax,nshells,npoints)
    rho_exp = [((t0/t)**3)*rho[i] for i in xrange(nshells+1)]
    return vbounds,rho_exp

def mix_abunds(velocities, t0,t,vmin,vmax,nshells,data,_runid, nepoch):
    #Hachinger Abundance Table
    fname= write_abunds_table(data, _runid, nepoch)
    abunds_hach=np.loadtxt(fname,skiprows=1)
    npoints=20
    #1/v grid
    vs,dens=homologous_expansion(t0,t,vmin,vmax,nshells,npoints)

    #boundaries of the 1/v grid, throw away vdummy
    vbounds=vs[1:]

    #abundances of 1/v grid shape: number of 1/v grid cells x number of elements +1 (index)
    abunds=np.zeros((len(vbounds),abunds_hach.shape[1]))

    for i in xrange(len(vbounds)-1):
        #vbounds[i] is left boundary of the i-th 1/v grid cell
        #vbounds[i+1] is right boundary of the i-th 1/v grid cell
        cell_mtot=0
        cell_abunds=np.zeros(abunds_hach.shape[1])
        for j in xrange(len(velocities)-1):
            #velocities[j] is the left interface of the j-th Hachinger grid cell
            #velocities[j+1] is the right interface of the j-th Hachinger grid cell

            if velocities[j+1]>vbounds[i]:
                #found first cell of Hachinger grid which falls into our 1/v grid cell
                #calculate mass in overlap region (density,tf*1e5, and 4/3pi cancel due to normalisation)
                cell_m=(np.min([velocities[j+1],vbounds[i+1]])**3-np.max([velocities[j],vbounds[i]])**3)
                cell_abunds+=cell_m*abunds_hach[j+1,:]
                cell_mtot+=cell_m

            if velocities[j+1]>=vbounds[i+1]:
                #stop - cell Hachinger is already in the next 1/v grid cell
                if cell_mtot==0:
                    #sth went wrong - abort!
                    print "Error: mtot zero in cell %d" %i
                    raise ValueError
                #normalize cell abundance
                cell_abunds/=cell_mtot
                #store final cell abundance in appropriate row of abundance data block
                abunds[i+1,:]=cell_abunds
                break

        #insert duplicate first line
        abunds[0,:]=abunds[1,:]

        #set indices of first column
        abunds[:,0]=np.arange(abunds.shape[0])

        #write abundance file
        f=open("abundances_%05d_%d.dat" % (_runid ,nepoch),"w")
        #write header
        f.write ("#Index Z=1 - Z=32\n")

        #save abundance data
        np.savetxt(f,abunds,fmt=["%2d"]+["%.4e" for i in xrange(abunds.shape[1]-1)])

        f.close()



def table_densities(t0,t,vmin,vmax,nshells,runid,nepoch):
    npoints=20
    vbounds,densities=homologous_expansion(t0,t,vmin,vmax,nshells,npoints)
    velocities=vbounds[1:]
    with open("densities_%05d_%d.dat" % (runid , nepoch),'w') as table:
        table.write("%1f %s\n" % (t, "day"))
        ind='#Index'
        vel='Velocities [km/s]'
        den='Densities [g/cm^3]'
        table.write("{0:<10s} {1:^10s} {2:>15s}\n".format(ind, vel, den))
        for i in xrange(nshells+1):
            table.write(
                "{0:>3d}{1:>10s}{2:>10.4e}{3:>10s}{4:>10e}\n".format(
                    i,
                    "".join([" " for _ in xrange(10)]),
                    velocities[i],
                    "".join([" " for _ in xrange(10)]),
                    densities[i],
                )
            )

def construct_W7_input_for_tardis(fname,velocities,t0,t,vmin,vmax, nshells,npoints):
    table_densities(t0,t,vmin,vmax, nshells,npoints)
    mix_abunds(fname,velocities,t0,t,vmin,vmax, nshells,npoints)


#if __name__ == "__main__":
#    table_densities(t0,t,vmin,vmax,nshells)
#    construct_W7_input_for_tardis("Abundances_m6_2SiS2CaMgchange.dat",(9000,15000,30000),0.000231481,11,9000,30000,20,20)
