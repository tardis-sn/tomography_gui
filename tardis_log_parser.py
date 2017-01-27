#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re

class plasma_strat_parser(object):
    def __init__(self, fstream):
        self.fstream = fstream

    def parse(self):

        buffer = self.fstream.readline()
        self.fstream.readline()
        tmp = buffer.rsplit()
        if tmp[-1] == "\\":
            # information spread over two blocks

            ncolumns = len(tmp)
            buffer = self.fstream.readline()
            tmp = buffer.rsplit()
            data = [tmp]
            while len(tmp) == ncolumns:
                data.append(tmp)
                buffer = self.fstream.readline()
                tmp = buffer.rsplit()
            tmp_data = np.array(data[1:], dtype = np.float)

            buffer = self.fstream.readline()
            self.fstream.readline()
            tmp = buffer.rsplit()
            try:
                assert(tmp[-1] != "\\")
            except AssertionError:
                print("Data spread over more than two blocks - no parser implementation for this case available")
                raise Exception
            ncolumns = len(tmp) + 1

            buffer = self.fstream.readline()
            tmp = buffer.rsplit()
            data = [tmp]
            while len(tmp) == ncolumns:
                data.append(tmp)
                buffer = self.fstream.readline()
                tmp = buffer.rsplit()
            self.data = np.append(tmp_data, np.array(data[1:], dtype = np.float)[:,1:], axis = 1)
        else:
            # information in one block
            ncolumns = len(tmp) + 1

            buffer = self.fstream.readline()
            tmp = buffer.rsplit()
            data = [tmp]
            while len(tmp) == ncolumns:
                data.append(tmp)
                buffer = self.fstream.readline()
                tmp = buffer.rsplit()
            self.data = np.array(data[1:], dtype = np.float)

        self.shells = self.data[:,0].astype(np.int)
        self.t_rads = self.data[:,1]
        self.next_t_rads = self.data[:,2]
        self.ws = self.data[:,3]
        self.new_ws = self.data[:,4]

class tardis_log_parser(object):

    def __init__(self, fname, mode):

        self.fname = fname
        #self.run_id = run_id
        self.fstream = open(fname, "r")
        self.cycles = []
        self.mode = mode
        self.parse()

    def parse(self):

        if self.mode == "bb":

            pattern = re.compile(".*K -- next t_inner.*")
            buffer = self.fstream.readline()

            j = 0
            t_inner = []
            while buffer != "":
                if j > 1000:
                    print("Something went wrong in outer loop; aborting")
                    break
                if re.match(pattern, buffer):
                    t_inner.append(float(buffer.rstrip().split('K -- next t_inner')[-1].split()[0]))
                buffer = self.fstream.readline()
                j+=1
            self.t_inner = t_inner
            print ("this is t_inner")
            print self.t_inner

        elif self.mode == "rad":

            pattern = re.compile(".*Plasma stratification.*")
            buffer = self.fstream.readline()

            j = 0
            t_inner = []
            while buffer != "":
                if j > 1000:
                    print("Something went wrong in outer loop; aborting")
                    break
                if re.match(pattern, buffer):
                    self.cycles.append(plasma_strat_parser(self.fstream))
                    self.cycles[-1].parse()
                buffer = self.fstream.readline()
                j+=1

            self.ws = np.array([c.ws for c in self.cycles])
            self.t_rads = np.array([c.t_rads for c in self.cycles])

    def visualise_convergence(self, fig = None):

        if self.mode == "bb":

            if fig is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)

            else:
                ax = fig.get_axes()[0]

            ax.plot(self.t_inner, label = r"$T_{\mathrm{inner}}$")
            ax.set_xlabel("Iterations")
            ax.set_ylabel(r"$T$ [K]")
            ax.legend(loc = "upper left")

        elif self.mode == "rad":

            if fig is None:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212, sharex = ax1)

            else:
                axes = fig.get_axes()
                ax1 = axes[0]
                ax2 = axes[1]

            [ax1.plot(tr, label= r"Shell %d" %i) for tr, i in zip(self.t_rads.T, self.cycles[0].shells)]
            ax1.set_ylabel(r"$T_{\mathrm{R}}$ [K]")

            [ax2.plot(ws) for ws, i in zip(self.ws.T, self.cycles[0].shells)]
            ax2.set_ylabel(r"$W$")
            ax2.set_xlabel("Iterations")
            ax1.legend(loc = "lower left", ncol = 3, prop = {"size":"small"})

        return fig

if __name__ == "__main__":

    tester = tardis_log_parser("logging.dat")
    tester.visualise_convergence()
    plt.show()
