#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
abunds_idents={"h":1,"he":2,"li":3,"be":4,"b":5,"c":6,"n":7,"o":8,"f":9,"ne":10,"na":11,"mg":12,"al":13,"si":14,"p":15,"s":16,"cl":17,"ar":18,"k":19,"ca":20,"sc":21,"ti":22,"v":23,"cr":24,"mn":25,"fe":26,"co":27,"ni":28,"cu":29,"zn":30,"ga":31,"ge":32}
ax = None

def ionization (mdl, species, Nshells):
    atn=abunds_idents[species]
    norm=[mdl.plasma_array.ion_populations[i][atn].values.sum() for i in xrange(Nshells)]
    ions=[mdl.plasma_array.ion_populations[i][atn].values / norm[i] for i in xrange(Nshells)]
    ions = np.array(ions)
    return ions

def ion_plot(mdl,fname, species, Nshells, fig):
    data = np.loadtxt(fname, skiprows=2)
    velocities = data[:,1]
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    else:
        ax = fig.get_axes()[0]
        ions = ionization(mdl, species, Nshells)
        for i in xrange(4):
            ax.plot(ions[:,i],velocities, label="%s %d" % (species, i))

        ax.set_ylabel("Ion Population")
        ax.set_xlabel("Velocities")
        ax.legend(prop={"size":"small"})
        ax.set_yscale('log')
