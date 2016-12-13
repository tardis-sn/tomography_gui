#!/usr/bin/env python
import numpy as np
import argparse
import glob
import shlex
import subprocess

def main(rootname):

    naepochs = glob.glob("tardis_%05d*.yml" % rootname)
    nepochs = len(naepochs)
    epochs =[]

    for i in xrange (nepochs):
        epoch =(naepochs[i].rsplit("_")[-1].strip()).rsplit(".")[-2].strip()
        print epoch
        epochs.append(int(epoch))

    return epochs

def write_submit(Nthreads, rootname, defaultfile = 'mpa-pascal-default.cmd'):

    epochs = main(rootname)

    for i in xrange(len(epochs)):

        filename = 'tardis_%05d_%d.yml' %(rootname, epochs[i])

        with open(defaultfile, 'r') as file:
            filedata = file.read()

        filedata = filedata.replace('NTHREADS', str(Nthreads))
        filedata = filedata.replace('executablefile', './run_tardis.py')
        filedata = filedata.replace('filename', str(filename))
        filedata = filedata.replace('rootname', str(rootname))
        filedata = filedata.replace('epoch', str(epochs[i]))

        with open('mpa-pascal_%05d_%d.cmd' % (rootname, epochs[i]), 'w') as file:
            file.write(filedata)

        p = subprocess.Popen(shlex.split("qsub mpa-pascal_%05d_%d.cmd" % (rootname, epochs[i])))

if __name__ == "__main__":
    write_submit(Nthreads, rootname)
