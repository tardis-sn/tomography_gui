#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import re


class plasma_strat_parser(object):
    def __init__(self, fstream, t_inner):
        self.fstream = fstream
        self.t_inner = t_inner

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


        self.fstream.readline()
        self.fstream.readline()
        self.t_inner_new = float(self.fstream.readline().rsplit("=")[-1].strip())

        self.shells = self.data[:,0].astype(np.int)
        self.converged_t_rads = self.data[:,1]
        self.converged_ws = self.data[:,2]
        self.new_trads = self.data[:,3]
        self.new_ws = self.data[:,4]
        self.t_rads = self.data[:,5]

        self.updated_t_rads = self.data[:,6]
        self.updated_ws = self.data[:,7]
        self.ws = self.data[:,8]

class tardis_log_parser(object):
    def __init__(self, fname):

        self.fname = fname
        self.fstream = open(fname, "r")
        self.cycles = []
        self.parse()

    def parse(self):

        pattern = re.compile(".*Plasma stratification.*")
        pattern_2 = re.compile(".*for t_inner=.*")
        buffer = self.fstream.readline()

        j = 0
        while buffer != "":
            if j > 1000:
                print("Something went wrong in outer loop; aborting")
                break
            i = 0
            while re.match(pattern, buffer) is None:
                if re.match(pattern_2, buffer):
                    t_inner = float(buffer.rsplit("=")[-1].strip())
                if i > 1000:
                    print("Something went wrong in inner loop; aborting")
                    break
                buffer = self.fstream.readline()
                if buffer == "":
                    break
                i+=1
            if buffer != "":
                print("Parsing Plasma State")
                self.cycles.append(plasma_strat_parser(self.fstream, t_inner))
                self.cycles[-1].parse()
            j+=1
            buffer = self.fstream.readline()

        self.ws = np.array([c.ws for c in self.cycles])
        self.t_rads = np.array([c.t_rads for c in self.cycles])
        self.t_inner = np.array([c.t_inner for c in self.cycles])
        self.t_inner_new = np.array([c.t_inner_new for c in self.cycles])

    def visualise_convergence(self, mode = "bb", fig = None):

        axes = None

        if fig is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        else:
            axes = fig.get_axes()
            ax1 = axes[0]
            ax2 = axes[1]

        if mode == "bb":
            ax1.plot(self.t_inner, label = r"$T_{\mathrm{BB}}$")
            ax1.plot(self.t_inner_new, label = r"$T_{\mathrm{BB}}^{\mathrm{new}}$")

            ax1.set_xlabel("iterations")
            ax1.set_ylabel(r"$T$ [K]")
            ax1.legend()

            ax2.plot(np.arange(1, len(self.t_inner)), (self.t_inner[1:] - self.t_inner[:-1]) / self.t_inner[1:])
            ax2.plot(np.arange(1, len(self.t_inner_new)), (self.t_inner_new[1:] - self.t_inner_new[:-1]) / self.t_inner_new[1:])
            ax2.set_xlabel("iterations")
            ax2.set_ylabel(r"relative change (previous step)")
            ax2.set_ylim([-0.05, 0.05])
        elif mode == "rad":

            pax1 = ax1.twinx()

            [ax1.plot(tr, label = r"$T_{\mathrm{R}}$, shell %d" % i) for tr, i in zip(self.t_rads.T, self.cycles[0].shells)]
            pax1.set_color_cycle(None)
            [pax1.plot(ws, label = r"$W$, shell %d" % i, ls = "dotted") for ws, i in zip(self.ws.T, self.cycles[0].shells)]


            ax1.set_xlabel("iterations")
            ax1.set_ylabel(r"$T$ [K]")
            pax1.set_ylabel(r"$W$")


            [ax2.plot(np.arange(1, len(self.t_inner)), (ws[1:] - ws[:-1]) / ws[1:], ls = "dotted") for ws, i in zip(self.ws.T, self.cycles[0].shells)]
            ax2.set_color_cycle(None)
            [ax2.plot(np.arange(1, len(self.t_inner)), (tr[1:] - tr[:-1]) / tr[1:], label = r"shell %d" % i) for tr, i in zip(self.t_rads.T, self.cycles[0].shells)]

            ax2.plot([0,0], [-10, -10], color = "black", ls = "solid", label = r"$T_{\mathrm{R}}$")
            ax2.plot([0,0], [-10, -10], color = "black", ls = "dotted", label = r"$W$")

            ax2.set_xlabel("iterations")
            ax2.set_ylabel(r"relative change (previous step)")
            ax2.set_ylim([-0.05, 0.05])
            ax2.legend(ncol = 3,prop = {"size":"small"})


        return fig


if __name__ == "__main__":

    tester = tardis_log_parser("logging.dat")
    tester.visualise_convergence()
    plt.show()
