#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def abundances_raw(fname,fig):
    data=np.loadtxt(fname,skiprows=2)
    abundsraw = data[:,6:]
    abundsraw=np.append(abundsraw,[abundsraw[-1,:]],axis=0)
    vmin=data[0,0]
    velocities=np.append(vmin,data[:,1])
    #velocities=(vel[1:]+vel[:-1])*0.5
    #abundsraw = abundsraw[::-1,:]
    #10 most important chemical elements: c(6), o(8), mg(12),si(14), s(16), ca(20),  ti(22), cr(24), fe (26), ni(28)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    else:
        ax = fig.get_axes()[0]
        [line.remove() for line in ax.get_lines()]

        ax.plot(velocities,abundsraw[:,5],color='r',label='C',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,7],color='b',label='O',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,11],color='k',label='Mg',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,13],color='c',label='Si',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,15],color='m',label='S',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,19],color='blueviolet',label='Ca',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,21],color='lime',label='Ti',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,23],color='y',label='Cr',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,25],color='g',label='Fe',drawstyle='steps-post')
        ax.plot(velocities,abundsraw[:,27],color='darkorange',label='Ni',drawstyle='steps-post')
        ax.set_ylabel("Abundances")
        ax.set_xlabel("Velocities")
        ax.legend(prop={"size":"small"})


def abundances_mix(fname,dfname,fig):
    abundsmix=np.loadtxt(fname,skiprows=1)
    data=np.loadtxt(dfname,skiprows=1)
    velocities=data[:,1]

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    else:
        ax = fig.get_axes()[0]
        mixed_lines = []
        mixed_lines.append(ax.plot(velocities,abundsmix[:,6],color='r',label='C',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,8],color='b',label='O',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,12],color='k',label='Mg',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,14],color='c',label='Si',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,16],color='m',label='S',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,20],color='blueviolet',label='Ca',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,22],color='lime',label='Ti',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,24],color='y',label='Cr',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,26],color='g',label='Fe',marker='+',linestyle='')[0])
        mixed_lines.append(ax.plot(velocities,abundsmix[:,28],color='darkorange',label='Ni',marker='+',linestyle='')[0])

    return mixed_lines
