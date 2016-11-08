#!/usr/bin/env python
import traceback
import os
import sys
import matplotlib
import matplotlib.pylab as plt
import yaml
import threading
import numpy as np
from math import factorial
from matplotlib.figure import *
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
from tardis import run_tardis
import logging
import StringIO
import tardis_log_parser_right as logparse
import time
import reddening as red
import create_input as cr
import abundances_plots as abplots
import lines_identification as lident
import tardis_gui as gui
import ionization_plot as ions

elements = { 'neut': 0, 'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8, 'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16, 'cl': 17, 'ar': 18, 'k': 19,    'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24, 'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32, 'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39,  'zr': 40, 'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48}
inv_elements = dict([(v,k) for k, v in elements.items()])

Zmax = 30
Nshellsfinal = 20
logging.basicConfig(filename="tardis_general.log", filemode = "w")

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

class inputwriterthread(QtCore.QThread):
    starttrigger = QtCore.pyqtSignal(int)
    endtrigger = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        super(inputwriterthread, self).__init__(parent)

        self.parent = parent

    def run(self):

        self.starttrigger.emit(0)

        try:
            self.parent.read_runid()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            self.endtrigger.emit(0)
            return False

        self.parent.save_abundance_file()
        self.endtrigger.emit(0)

class tardisthread(QtCore.QThread):
    starttrigger = QtCore.pyqtSignal(int)
    endtrigger = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        super(tardisthread, self).__init__(parent)

        self.parent = parent

    def run(self):

        self.starttrigger.emit(0)

        try:
            self.parent.read_runid()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            self.endtrigger.emit(1)
            return False

        try:

            startpos = logging.getLogger().handlers[0].stream.tell()
            mdl = run_tardis(self.parent.tardis_config)
            mdl.save_spectra("spec_%05d.dat" % self.parent.runid)
            if self.parent.save_model:
                mdl.to_hdf5("model_%05d.h5" % self.parent.runid)
                mdl.atom_data.lines.to_hdf("lines_%05d.h5" % self.parent.runid, "lines")

            endpos = logging.getLogger().handlers[0].stream.tell()
            logging.getLogger().handlers[0].stream.flush()

            f = open("tardis_general.log", "r")
            f.seek(startpos)
            log = f.read(endpos-startpos)
            modellog = open("tardis_%05d.log" % self.parent.runid, "w")
            modellog.write(log)
            modellog.close()
            f.close()

            # @Michi: when running tardis via this gui, the model is stored at
            # the end in the gui (in order to be able to use the old tardis gui
            # to do some additional diagnostics on the model). Could it be that
            # the python garbage collector does not clean up the old tardis
            # model whenever a new run is started? If this is the case, how
            # could we get around it? With something like:
            # del(self.parent.mdl)
            # self.parent.mdl = mdl
            # @Talytha: as an ugly quickfix, just comment the following line
            self.parent.mdl = mdl
            logging.getLogger().handlers[0].stream.seek(0)
            self.endtrigger.emit(0)
            print("Tardis run done")

        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)
            print(ex)
            print("Warning: tardis run failed")
            self.endtrigger.emit(1)
            return False

class MatplotlibWidget(FigureCanvas):

    def __init__(self, parent, fig=None):

        self.parent = parent
        self.figure = Figure()
        self.cid = {}
        if fig is None:
            self.ax = self.figure.add_subplot(111)
        elif fig == "convergence":
            self.ax = [self.figure.add_subplot(211), self.figure.add_subplot(212)]
        self.cb = None
        self.span = None

        super(MatplotlibWidget, self).__init__(self.figure)
        super(MatplotlibWidget, self).setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        super(MatplotlibWidget, self).updateGeometry()

        self.toolbar = NavigationToolbar(self, parent)

class Example(QtGui.QWidget):

    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def initUI(self):

        self.current_data = None
        self.old_data = None
        self.obs_spectrum  = None
        self.current_spectrum = None
        self.old_spectrum = None
        self.reddening = None
        self.distance_modulus = None
        self.tardis_config = None
        self.tardis_running = 0
        self.rescale_model = False
        self.save_model = False
        self.redden_model = False
        self.virtual_model = False
        self.show_oldrun = False
        self.filter_model = False
        self.mdl = None
        self.mixed_lines = None
        #self.raw_abund_data = None
        self.observation_data = None
        self.lamax = None
        self.lamin = None
        self.nepochabundances = None
        self.nepochlines = None
        self.nepochplot = None
        self.ion = None
        self.addshell_index = 0
        self.removeshell_index = 0
        self.window = None

        self.table = QtGui.QTableWidget(5,Zmax+7,self)
        self.table.setHorizontalHeaderLabels(["active", "Vmin", "Vmax", "t", "logL/Lsun", "lam min", "lam max"] + [inv_elements[z].capitalize() for z in xrange(1, Zmax+1)])
        self.addshell_entry = QtGui.QLineEdit(self)
        self.removeshell_entry = QtGui.QLineEdit(self)

        self.runid_entry = QtGui.QLineEdit(self)
        self.oldrunid_entry = QtGui.QLineEdit(self)
        self.runidplot_entry = QtGui.QLineEdit(self)
        self.oldrunidplot_entry = QtGui.QLineEdit(self)
        self.nepochplot_entry = QtGui.QLineEdit(self)

        self.risetime_entry = QtGui.QLineEdit(self)

        self.distance_entry = QtGui.QLineEdit(self)
        self.reddening_entry = QtGui.QLineEdit(self)
        self.window_entry = QtGui.QLineEdit(self)
        self.savemodel_cbox = QtGui.QCheckBox("Save model", self)
        self.showoldrun_cbox = QtGui.QCheckBox("Show previous run", self)
        self.filtermodel_cbox = QtGui.QCheckBox("Apply Savitzky Golay filter", self)

        self.appendshell_button = QtGui.QPushButton("Append Shell")
        self.addshell_button = QtGui.QPushButton("Add Shell")
        self.removeshell_button = QtGui.QPushButton("Remove Shell")
        self.createinput_button = QtGui.QPushButton("Create Input")
        self.loadabundances_button = QtGui.QPushButton("Load Abundances")
        self.saveabundances_button = QtGui.QPushButton("Save Abundances and Tardis Files")
        self.runtardis_button = QtGui.QPushButton("Run Tardis")
        self.loadobservation_button = QtGui.QPushButton("Load Observation")
        self.updateplots_button = QtGui.QPushButton("Update Plots")
        self.showgui_button = QtGui.QPushButton("Tardis Gui")
        self.virtualmodel_cbox = QtGui.QCheckBox("Show Virtual Spectrum", self)
        self.rescalemodel_cbox = QtGui.QCheckBox("Apply Distance Modulus", self)
        self.reddenmodel_cbox = QtGui.QCheckBox("Apply Reddening", self)
        self.clearplot_button = QtGui.QPushButton("Clear Figure")

        self.radiationfieldconvergence_button = QtGui.QPushButton("Show Radiation Field Convergence")
        self.bbconvergence_button = QtGui.QPushButton("Show Black-Body Convergence")
        self.runidconvergence_entry = QtGui.QLineEdit(self)

        self.abundancesraw_button=QtGui.QPushButton("Raw Abundances")
        self.abundancesmix_cbox = QtGui.QCheckBox("Mixed Abundances", self)
        self.runidabundances_entry= QtGui.QLineEdit(self)
        self.nepochabundances_entry = QtGui.QLineEdit(self)

        self.trads_button=QtGui.QPushButton("Radiation Temperatures")
        self.ws_button=QtGui.QPushButton("Dilution Factors")
        self.runidtrads_ws_entry=QtGui.QLineEdit(self)

        self.lineshist_button=QtGui.QPushButton("Last Element Contribution")
        self.lineskromer_button=QtGui.QPushButton("Kromer Plot")
        self.runidlines_entry=QtGui.QLineEdit(self)
        self.lamin_entry=QtGui.QLineEdit(self)
        self.lamax_entry=QtGui.QLineEdit(self)
        self.nepochlines_entry = QtGui.QLineEdit(self)

        self.ion_button=QtGui.QPushButton("Ionization Plot")
        self.runidion_entry=QtGui.QLineEdit(self)
        self.ion_entry=QtGui.QLineEdit(self)

        self.addshell_entry.setText("0")
        self.removeshell_entry.setText("0")

        self.spectrum_figure = MatplotlibWidget(self)
        self.convergence_figure = MatplotlibWidget(self, fig = "convergence")
        self.abundances_figure= MatplotlibWidget(self)
        self.trads_ws_figure=MatplotlibWidget(self, fig = "convergence")
        self.lines_figure=MatplotlibWidget(self, fig= "convergence")
        self.ion_figure=MatplotlibWidget(self)

        table_hbox = QtGui.QHBoxLayout()
        table_hbox.addWidget(self.table)

        abundance_control_grid = QtGui.QGridLayout()

        abundance_control_grid.addWidget(QtGui.QLabel("Rise time:"), 0, 0)
        abundance_control_grid.addWidget(self.risetime_entry, 0, 1)
        abundance_control_grid.addWidget(QtGui.QLabel("Model: Run ID"), 0, 2)
        abundance_control_grid.addWidget(self.runid_entry, 0, 3)
        abundance_control_grid.addWidget(QtGui.QLabel("Model: Old run ID"), 0, 4)
        abundance_control_grid.addWidget(self.oldrunid_entry, 0, 5)

        abundance_control_grid.addWidget(self.loadabundances_button, 1, 0)
        abundance_control_grid.addWidget(self.appendshell_button, 2, 0)
        abundance_control_grid.addWidget(self.removeshell_button, 2, 1)
        abundance_control_grid.addWidget(self.removeshell_entry, 3, 1)
        abundance_control_grid.addWidget(self.addshell_button, 2, 3)
        abundance_control_grid.addWidget(self.addshell_entry, 3, 3)
        abundance_control_grid.addWidget(self.saveabundances_button, 4, 0)
        abundance_control_grid.addWidget(self.runtardis_button, 5, 0)
        abundance_control_grid.addWidget(self.savemodel_cbox, 5, 1)

        spectrum_control_grid = QtGui.QGridLayout()

        spectrum_control_grid.addWidget(QtGui.QLabel("Plotting: Run ID"), 0, 0)
        spectrum_control_grid.addWidget(self.runidplot_entry, 0, 1)
        spectrum_control_grid.addWidget(QtGui.QLabel("Plotting: Old run ID"), 0, 2)
        spectrum_control_grid.addWidget(self.oldrunidplot_entry, 0, 3)
        spectrum_control_grid.addWidget(QtGui.QLabel("Epoch"), 0, 4)
        spectrum_control_grid.addWidget(self.nepochplot_entry, 0 ,5)

        spectrum_control_grid.addWidget(self.loadobservation_button, 1, 0)
        spectrum_control_grid.addWidget(self.updateplots_button, 1, 1)
        spectrum_control_grid.addWidget(self.clearplot_button, 1, 2)
        spectrum_control_grid.addWidget(self.showgui_button, 1, 3)

        spectrum_control_grid.addWidget(self.virtualmodel_cbox, 2, 0)
        spectrum_control_grid.addWidget(self.showoldrun_cbox, 2, 1)
        spectrum_control_grid.addWidget(self.filtermodel_cbox, 2, 2)
        spectrum_control_grid.addWidget(self.rescalemodel_cbox, 2, 3)
        spectrum_control_grid.addWidget(self.reddenmodel_cbox, 2, 4)

        spectrum_control_grid.addWidget(QtGui.QLabel("Distance modulus"), 3, 0)
        spectrum_control_grid.addWidget(self.distance_entry, 3, 1)
        spectrum_control_grid.addWidget(QtGui.QLabel("Reddening E(B-V)"), 3, 2)
        spectrum_control_grid.addWidget(self.reddening_entry, 3, 3)
        spectrum_control_grid.addWidget(QtGui.QLabel("Filter (Window Size)"), 3, 4)
        spectrum_control_grid.addWidget(self.window_entry, 3, 5)

        convergence_control_grid = QtGui.QGridLayout()

        convergence_control_grid.addWidget(QtGui.QLabel("Convergence: Run ID"), 0, 0)
        convergence_control_grid.addWidget(self.runidconvergence_entry, 0, 1)
        convergence_control_grid.addWidget(self.bbconvergence_button, 1, 0)
        convergence_control_grid.addWidget(self.radiationfieldconvergence_button, 1, 1)

        abundances_control_grid= QtGui.QGridLayout()

        abundances_control_grid.addWidget(QtGui.QLabel("Abundances: Run ID"), 0, 0)
        abundances_control_grid.addWidget(self.runidabundances_entry,0,1)
        abundances_control_grid.addWidget(QtGui.QLabel("Epoch"), 0, 2)
        abundances_control_grid.addWidget(self.nepochabundances_entry,0,3)
        abundances_control_grid.addWidget(self.abundancesraw_button,1,0)
        abundances_control_grid.addWidget(self.abundancesmix_cbox,1,1)

        trads_ws_control_grid= QtGui.QGridLayout()

        trads_ws_control_grid.addWidget(QtGui.QLabel("Trads Ws Run: ID"),0,0)
        trads_ws_control_grid.addWidget(self.runidtrads_ws_entry,0,1)
        trads_ws_control_grid.addWidget(self.trads_button,1,0)
        trads_ws_control_grid.addWidget(self.ws_button,1,1)

        lines_control_grid= QtGui.QGridLayout()

        lines_control_grid.addWidget(QtGui.QLabel("Lines: Run ID"),0,0)
        lines_control_grid.addWidget(self.runidlines_entry,0,1)
        lines_control_grid.addWidget(QtGui.QLabel("Lambda Min"),0,2)
        lines_control_grid.addWidget(self.lamin_entry,0,3)
        lines_control_grid.addWidget(QtGui.QLabel("Lambda Max"),0,4)
        lines_control_grid.addWidget(self.lamax_entry,0,5)
        lines_control_grid.addWidget(QtGui.QLabel("Epoch"), 0,6)
        lines_control_grid.addWidget(self.nepochlines_entry,0,7)
        lines_control_grid.addWidget(self.lineshist_button,1,0)
        lines_control_grid.addWidget(self.lineskromer_button,1,1)

        ion_control_grid = QtGui.QGridLayout()

        ion_control_grid.addWidget(QtGui.QLabel("Ionization: Run ID"),0,0)
        ion_control_grid.addWidget(self.runidion_entry,0,1)
        ion_control_grid.addWidget(QtGui.QLabel("Name of the Element"),0,2)
        ion_control_grid.addWidget(self.ion_entry,0,3)
        ion_control_grid.addWidget(self.ion_button,0,4)


        abundance_vbox = QtGui.QVBoxLayout()
        abundance_vbox.addLayout(table_hbox)
        abundance_vbox.addLayout(abundance_control_grid)

        diagnostics_tab_widget = QtGui.QTabWidget()
        spectrum_tab = QtGui.QWidget()
        spectrum_vbox = QtGui.QVBoxLayout(spectrum_tab)
        spectrum_vbox.addWidget(self.spectrum_figure)
        spectrum_vbox.addWidget(self.spectrum_figure.toolbar)
        spectrum_vbox.addLayout(spectrum_control_grid)
        diagnostics_tab_widget.addTab(spectrum_tab, "Spectrum")

        convergence_tab = QtGui.QWidget()
        convergence_vbox = QtGui.QVBoxLayout(convergence_tab)
        convergence_vbox.addWidget(self.convergence_figure)
        convergence_vbox.addWidget(self.convergence_figure.toolbar)
        convergence_vbox.addLayout(convergence_control_grid)
        diagnostics_tab_widget.addTab(convergence_tab, "Convergence")

        abundances_tab= QtGui.QWidget()
        abundances_vbox= QtGui.QVBoxLayout(abundances_tab)
        abundances_vbox.addWidget(self.abundances_figure)
        abundances_vbox.addWidget(self.abundances_figure.toolbar)
        abundances_vbox.addLayout(abundances_control_grid)
        diagnostics_tab_widget.addTab(abundances_tab, "Abundances")

        trads_ws_tab= QtGui.QWidget()
        trads_ws_vbox= QtGui.QVBoxLayout(trads_ws_tab)
        trads_ws_vbox.addWidget(self.trads_ws_figure)
        trads_ws_vbox.addWidget(self.trads_ws_figure.toolbar)
        trads_ws_vbox.addLayout(trads_ws_control_grid)
        diagnostics_tab_widget.addTab(trads_ws_tab, "Trads and Ws")

        lines_tab= QtGui.QWidget()
        lines_vbox= QtGui.QVBoxLayout(lines_tab)
        lines_vbox.addWidget(self.lines_figure)
        lines_vbox.addWidget(self.lines_figure.toolbar)
        lines_vbox.addLayout(lines_control_grid)
        diagnostics_tab_widget.addTab(lines_tab, "Lines Identification")

        ion_tab= QtGui.QWidget()
        ion_vbox= QtGui.QVBoxLayout(ion_tab)
        ion_vbox.addWidget(self.ion_figure)
        ion_vbox.addWidget(self.ion_figure.toolbar)
        ion_vbox.addLayout(ion_control_grid)
        diagnostics_tab_widget.addTab(ion_tab, "Ionization")


        main_hbox = QtGui.QHBoxLayout()
        main_hbox.addWidget(diagnostics_tab_widget)
        main_hbox.addLayout(abundance_vbox)

        self.setLayout(main_hbox)

        self.addshell_entry.textChanged[str].connect(self.addshell_entry_changed)
        self.removeshell_entry.textChanged[str].connect(self.removeshell_entry_changed)
        self.oldrunid_entry.textChanged[str].connect(self.oldrunid_entry_changed)
        self.runid_entry.textChanged[str].connect(self.runid_entry_changed)
        self.nepochplot_entry.textChanged[str].connect(self.nepochplot_entry_changed)
        self.risetime_entry.textChanged[str].connect(self.risetime_entry_changed)
        self.addshell_button.clicked.connect(self.on_addshell_clicked)
        self.removeshell_button.clicked.connect(self.on_removeshell_clicked)
        self.appendshell_button.clicked.connect(self.on_appendshell_clicked)
        self.loadabundances_button.clicked.connect(self.load_abundance_file)
        self.saveabundances_button.clicked.connect(self.save_input_files)
        self.loadobservation_button.clicked.connect(self.load_observation_file)
        self.bbconvergence_button.clicked.connect(self.plot_bb_convergence)
        self.radiationfieldconvergence_button.clicked.connect(self.plot_rad_convergence)
        self.abundancesraw_button.clicked.connect(self.plot_abundances_raw)
        self.abundancesmix_cbox.stateChanged.connect(self.abundancesmix_changed)

        #self.trads_button.clicked.connect(self.plot_trads)
        #self.ws_button.clicked.connect(self.plot_ws)
        self.lineshist_button.clicked.connect(self.plot_lineshist)
        self.lineskromer_button.clicked.connect(self.plot_lineskromer)
        self.ion_button.clicked.connect(self.plot_ion)

        self.runtardis_button.clicked.connect(self.start_tardis)
        self.updateplots_button.clicked.connect(self.update_plots)
        self.distance_entry.textChanged[str].connect(self.distance_entry_changed)
        self.reddening_entry.textChanged[str].connect(self.reddening_entry_changed)
        self.window_entry.textChanged[str].connect(self.window_entry_changed)
        self.savemodel_cbox.stateChanged.connect(self.savemodel_changed)
        self.rescalemodel_cbox.stateChanged.connect(self.rescalemodel_changed)
        self.reddenmodel_cbox.stateChanged.connect(self.reddenmodel_changed)
        self.virtualmodel_cbox.stateChanged.connect(self.virtualmodel_changed)
        self.showoldrun_cbox.stateChanged.connect(self.showoldrun_changed)
        self.filtermodel_cbox.stateChanged.connect(self.filtermodel_changed)
        self.runidplot_entry.textChanged[str].connect(self.runidplot_entry_changed)
        self.oldrunidplot_entry.textChanged[str].connect(self.oldrunidplot_entry_changed)
        self.showgui_button.clicked.connect(self.show_gui)
        self.runidconvergence_entry.textChanged[str].connect(self.runidconvergence_entry_changed)
        self.runidabundances_entry.textChanged[str].connect(self.runidabundances_entry_changed)
        self.nepochabundances_entry.textChanged[str].connect(self.nepochabundances_entry_changed)
        self.nepochlines_entry.textChanged[str].connect(self.nepochlines_entry_changed)
        self.runidtrads_ws_entry.textChanged[str].connect(self.runidtrads_ws_entry_changed)
        self.runidlines_entry.textChanged[str].connect(self.runidlines_entry_changed)
        self.lamin_entry.textChanged[str].connect(self.lamin_entry_changed)
        self.lamax_entry.textChanged[str].connect(self.lamax_entry_changed)
        self.runidion_entry.textChanged[str].connect(self.runidion_entry_changed)
        self.ion_entry.textChanged[str].connect(self.ion_entry_changed)
        self.clearplot_button.clicked.connect(self.clear_plot)

        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Tardis Abundance Tomography')
        self.show()

    def show_gui(self):

        if self.mdl is None:
            print("Warning: no model available")
            return False

        mygui = gui.ModelViewer()
        mygui.show_model(self.mdl)

        #mygui = gui.Tardis()
        #mygui.show_model(self.mdl)

    def filtermodel_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.filter_model = True
        else:
            self.filter_model = False

    def showoldrun_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.show_oldrun = True
        else:
            self.show_oldrun = False
            self.old_spectrum.remove()
            self.old_spectrum = None
            self.old_data = None
            #self.spectrum_figure.figure.canvas.draw()

    def abundancesmix_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.plot_abundances_mix()

        else:
            [line.remove() for line in self.mixed_lines]
            self.mixed_lines = None
            self.abundances_figure.figure.canvas.draw()

    def virtualmodel_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.virtual_model = True
        else:
            self.virtual_model = False

    def reddenmodel_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.redden_model = True
        else:
            self.redden_model = False

    def savemodel_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.save_model = True
        else:
            self.save_model = False

    def rescalemodel_changed(self, state):

        if state == QtCore.Qt.Checked:
            self.rescale_model = True
        else:
            self.rescale_model = False

    def savingfiles_started(self, sig):

        print("Saving files started")
        self.runtardis_button.setEnabled(False)

    def savingfiles_ended(self, sig):

        print("Saving files ended")
        self.runtardis_button.setEnabled(True)

    def read_reddening(self):

        try:
            reddening = float(self.reddeningtext)
        except ValueError:
            print("Warning: invalid reddening '%s'" % self.reddeningtext)
            raise Exception

        self.reddening = reddening

    def read_window(self):

        try:
            window = int(self.windowtext)
        except ValueError:
            print("Warning: invalid reddening '%s'" % self.windowtext)
            raise Exception

        self.window = window

    def read_lamin(self):

        try:
            lamin = float(self.lamintext)
        except ValueError:
            print("Warning: invalid minimum value '%s' for lambda" % self.lamintext)
            raise Exception

        self.lamin = lamin

    def read_lamax(self):

        try:
            lamax = float(self.lamaxtext)
        except ValueError:
            print("Warning: invalid maximum value '%s' for lambda" % self.lamaxtext)
            raise Exception

        self.lamax = lamax

    def read_risetime(self):

        try:
            risetime = float(self.risetimetext)
        except ValueError:
            print("Warning: invalid risetime '%s'" % self.risetimetext)
            raise Exception

        self.risetime = risetime

    def read_runid(self):

        try:
            runid = int(self.runidtext)
        except ValueError:
            print("Warning: invalid runid '%s'" % self.runidtext)
            raise Exception

        try:
            oldrunid = int(self.oldrunidtext)
        except ValueError:
            print("Warning: invalid old runid '%s'" % self.oldrunidtext)
            raise Exception

        self.runid = runid
        self.oldrunid = oldrunid

    def read_nepochplot(self):

       try:
           nepochplot = int(self.nepochplottext)
       except ValueError:
           print ("Warning: invalid epoch '%s'" % nepochplottext)
           raise Exception

       self.nepochplot = nepochplot


    def read_runidplot(self):


        try:
            runid = int(self.runidplottext)
        except ValueError:
            print("Warning: invalid runid '%s'" % self.runidplottext)
            raise Exception

        try:
            oldrunid = int(self.oldrunidplottext)
        except ValueError:
            print("Warning: invalid old runid '%s'" % self.oldrunidplottext)
            raise Exception

        self.runidplot = runid
        self.oldrunidplot = oldrunid


    def read_runidconvergence(self):

        try:
            runid = int(self.runidconvergencetext)
        except ValueError:
            print("Warning: invalid runid '%s'" % runidconvergencetext)
            raise Exception

        self.runidconvergence = runid


    def read_runidabundances(self):

       try:
           runid = int(self.runidabundancestext)
       except ValueError:
           print ("Warning: invalid runid '%s'" % runidabundancestext)
           raise Exception

       self.runidabundances = runid

    def read_nepochabundances(self):

       try:
           nepochabundances = int(self.nepochabundancestext)
       except ValueError:
           print ("Warning: invalid epoch '%s'" % nepochabundancestext)
           raise Exception

       self.nepochabundances = nepochabundances

    def read_nepochlines(self):

       try:
           nepochlines = int(self.nepochlinestext)
       except ValueError:
           print ("Warning: invalid epoch '%s'" % nepochlinestext)
           raise Exception

       self.nepochlines = nepochlines

    def read_runidtrads_ws(self):

        try:
            runid = int(self.runidtrads_wstext)
        except ValueError:
            print("Warning: invalid runid '%s'" % runidtrads_wstext)

            raise Exception

        self.runidtrads_ws = runid

    def read_runidlines(self):

        try:
            runid = int(self.runidlinestext)
        except ValueError:
            print("Warning: invalid runid '%s'" % runidlinestext)

            raise Exception

        self.runidlines = runid

    def read_runidion(self):

        try:
            runid = int(self.runidiontext)
        except ValueError:
            print("Warning: invalid runid '%s'" % runidiontext)

            raise Exception

        self.runidion = runid

    def read_ion(self):

        try:
            ion = str(self.iontext)
        except ValueError:
            print("Warning: invalid runid '%f'" % iontext)

            raise Exception

        self.ion = ion


    def start_tardis(self):

        try:
            self.read_runid()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        fname = "tardis_%05d.yml" % self.runid
        try:
            self.tardis_config = yaml.safe_load(open(fname, "r"))
        except IOError:
            print("Warning: could not open Tardis config '%s'" % fname)
            return False

            ##self.oldlogger = logging.getLogger
        thread = tardisthread(self)
        thread.starttrigger.connect(self.tardis_started)
        thread.endtrigger.connect(self.tardis_ended)
        thread.start()


    def tardis_started(self, sig):

        self.runtardis_button.setEnabled(False)

    def tardis_ended(self, sig):

        self.runtardis_button.setEnabled(True)
        if sig == 0:
            self.finish_tardis_run()

    def finish_tardis_run(self):

        try:
            self.read_runid()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        self.runidplot_entry.setText(str(self.runid))
        self.oldrunidplot_entry.setText(str(self.oldrunid))

        self.runidconvergence_entry.setText(str(self.runid))
        self.runidabundances_entry.setText(str(self.runid))
        #self.runidtrads_ws_entry.setText(str(self.runid))
        self.runidlines_entry.setText(str(self.runid))
        self.runidion_entry.setText(str(self.runid))

        if self.current_data is not None:
            self.old_data = self.current_data

        if not self.update_plots():
            print("Warning: Updating Plots after Tardis Run Failed")
            return False

        self.oldrunid = self.runid
        self.runid = self.runid + 1

        self.oldrunid_entry.setText(str(self.oldrunid))
        self.runid_entry.setText(str(self.runid))

    def clear_plot(self):

        self.spectrum_figure.ax.clear()
        self.current_spectrum = None
        self.old_spectrum = None
        self.spectrum_figure.figure.canvas.draw()


    def update_plots(self):

        try:
            self.read_runidplot()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        try:
                self.read_nepochplot()

        except ValueError:
                print("Warning: no epoch specified")
                return False


        if self.show_oldrun:

            if self.virtual_model:
                fname = "virtual_spec_%05d_%d.dat" % (self.oldrunidplot, self.nepochplot)
            else:
                fname = "spec_%05d_%d.dat" % (self.oldrunidplot, self.nepochplot)

            try:
                f = open(fname, "r")
            except IOError:
                print("Warning: no appropriate spectrum file was found")
                return False

            self.old_data = np.loadtxt(fname)

        if self.virtual_model:
            fname = "virtual_spec_%05d_%d.dat" % (self.runidplot, self.nepochplot)
        else:
            fname = "spec_%05d_%d.dat" % (self.runidplot, self.nepochplot)

        try:
            f = open(fname, "r")
        except IOError:
            print("Warning: no appropriate spectrum file was found")
            return False

        self.current_data = np.loadtxt(fname)

        self.plot_model()
        self.plot_observation()
        self.spectrum_figure.figure.canvas.draw()

        return True

    def plot_bb_convergence(self):

        self.plot_convergence(mode = "bb")

    def plot_rad_convergence(self):

        self.plot_convergence(mode = "rad")

    def plot_convergence(self, mode = "bb"):

        try:
            self.read_runidconvergence()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        axes = self.convergence_figure.figure.get_axes()
        [ax.clear() for ax in self.convergence_figure.figure.get_axes()]

        parser = logparse.tardis_log_parser("tardis_%05d.log" % self.runidconvergence)
        parser.visualise_convergence(mode = mode, fig = self.convergence_figure.figure)

        self.convergence_figure.figure.canvas.draw()

    def plot_ion(self):

        try:
            self.read_runidion()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False


        if self.mdl is None:
                print("Warning: no model available")
                return False


        try:
            self.read_ion()
        except ValueError:
            print("Warning: no element selected")
            return False

        #check = False
        for k, v in elements.items():
            if self.ion == k:
                #check = True
                self.ion = k

        ax = self.ion_figure.figure.gca()

        if self.ion is not None:
            [line.remove() for line in ax.get_lines()]
            ions.ion_plot(self.mdl, self.ion, Nshellsfinal, fig= self.ion_figure.figure)

        else:
                print ("No existing element")
                return False

        ax.autoscale_view(True,True,True)
        self.ion_figure.figure.canvas.draw()

    def plot_lineshist(self):

        self.plot_lines(mode = "ht")

    def plot_lineskromer(self):

        self.plot_lines(mode = "kr")

    def plot_lines(self, mode = "ht"):

        try:
            self.read_runidlines()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        axes = self.lines_figure.figure.get_axes()

        if mode == 'ht':

            #if self.mdl is None:
            #    print("Warning: no model available")
            #    return False

            try:
                self.read_lamin()
            except ValueError:
                print("Warning: no minimum value for lambda")
                return False

            try:
                self.read_lamax()
            except ValueError:
                print("Warning: no maximum value for lambda")
                return False

            try:
                self.read_nepochlines()

            except ValueError:
                print("Warning: no epoch specified")
                return False


            model= "model_%05d_%d.h5" % (self.runidlines, self.nepochlines)
            lines = "lines_%05d_%d.h5" % (self.runidlines, self.nepochlines)

            ax1=axes[0]
            ax1.clear()
            #[ax1.clear() for ax1 in self.lines_figure.figure.get_axes()]
            lident.lineshist(model, lines, self.lamin, self.lamax,fig = self.lines_figure.figure)
            ax1.autoscale_view(True,True,True)

        elif mode == 'kr':

            #if self.mdl is None:
                #print("Warning: no model available")
                #return False

            try:
                self.read_lamin()
            except ValueError:
                print("Warning: no minimum value for lambda")
                return False

            try:
                self.read_lamax()
            except ValueError:
                print("Warning: no maximum value for lambda")
                return False

            try:
                self.read_nepochlines()

            except ValueError:
                print("Warning: no epoch specified")
                return False


            if len(axes)==3:
                self.lines_figure.figure.delaxes(self.lines_figure.figure.get_axes()[-1])
            self.lines_figure.figure.delaxes(self.lines_figure.figure.get_axes()[-1])
            self.lines_figure.figure.add_subplot(212)

            model= "model_%05d_%d.h5" % (self.runidlines, self.nepochlines)
            lines = "lines_%05d_%d.h5" % (self.runidlines, self.nepochlines)

            axes = self.lines_figure.figure.get_axes()
            ax2=axes[1]
            ax2.clear()
            #[ax2.clear() for ax2 in self.lines_figure.figure.get_axes()]
            lident.lineskromer(model, lines, self.lamin, self.lamax, fig = self.lines_figure.figure)
            ax2.autoscale_view(True,True,True)
        self.lines_figure.figure.canvas.draw()

    #def plot_trads(self):

     #   self.plot_trads_ws(mode="trads")

    #def plot_ws(self):

     #   self.plot_trads_ws(mode="ws")

    def plot_abundances_raw(self):

        try:
            self.read_runidabundances()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print ("Warning: could not determine runids")
            return False

        try:
                self.read_nepochabundances()
        except ValueError:
                print("Warning: no epoch specified")
                return False

        #if self.raw_abund_data is not None:
        ax = self.abundances_figure.figure.gca()
        [line.remove() for line in ax.get_lines()]
        fname = "abundances_raw_%05d_%d.txt" % (self.runidabundances, self.nepochabundances)
        abplots.abundances_raw(fname,fig = self.abundances_figure.figure)
        ax.autoscale_view(True,True,True)
        self.abundances_figure.figure.canvas.draw()

        #else:
        #    print ("Warning: Abundances were not saved")
        #    return False

    def plot_abundances_mix(self):

        try:
            self.read_runidabundances()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print ("Warning: could not determine runids")
            return False

        try:
                self.read_nepochabundances()
        except ValueError:
                print("Warning: no epoch specified")
                return False

        ax= self.abundances_figure.figure.gca()

        if self.mixed_lines is None:
            fname="abundances_%05d_%d.dat" % (self.runidabundances, self.nepochabundances)
            dfname="densities_%05d_%d.dat" % (self.runidabundances, self.nepochabundances)
            self.mixed_lines=abplots.abundances_mix(fname,dfname,fig=self.abundances_figure.figure)

        else:
            [line.remove() for line in self.mixed_lines]
            self.mixed_lines = None

        ax.autoscale_view(True,True,True)
        self.abundances_figure.figure.canvas.draw()

    def plot_model(self):

        try:
            self.read_runidplot()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        if self.rescale_model:
            if self.distance_modulus is None:
                print("Warning: distance modulus not set")
                return False
            try:
                distance_modulus = float(self.distance_modulus)
            except ValueError:
                print("Warning: invalid distance modulus '%s'" % self.distance_modulus)
                return False
            distance = 10**((distance_modulus - 25.) / 5.) * 1e6 * 3.0857e18
            L_to_F = 4. * np.pi * distance**2

        else:

            L_to_F = 1

        if self.redden_model:

            try:
                self.read_reddening()

            except ValueError:
                print("Warning: no reddening")
                return False

            if self.old_data is not None:
                reddening_old, mask_old = red.calculate_reddening_correction(self.old_data[:,0], self.old_data[:,1],self.reddening)
                self.old_data = self.old_data[mask_old,:]
            else:
                reddening_old = 1

            reddening_current, mask_current = red.calculate_reddening_correction(self.current_data[:,0], self.current_data[:,1], self.reddening)
            self.current_data = self.current_data[mask_current,:]

        else:

            reddening_old = 1
            reddening_current = 1

        ax = self.spectrum_figure.figure.gca()
        if self.old_data is not None:
            x = self.old_data[:,0]
            y = self.old_data[:,1] / L_to_F * reddening_old
            if self.filter_model:
                try:
                    self.read_window()

                except ValueError:
                    print("Warning: no window size specified")
                    return False
                #window_size=31
                y = savitzky_golay(y, window_size= self.window, order=4)
            if self.old_spectrum is None:
                self.old_spectrum = ax.plot(x, y, color = "red")[0]
            else:
                self.old_spectrum.set_data(x, y)
            self.old_spectrum.set_label("Run: %05d" % self.oldrunidplot)

        if self.current_data is not None:
            x = self.current_data[:,0]
            y = self.current_data[:,1] / L_to_F * reddening_current
            if self.filter_model:
                try:
                    self.read_window()

                except ValueError:
                    print("Warning: no window size specified")
                    return False

                y = savitzky_golay(y, window_size=self.window, order=4)

            if self.current_spectrum is None:
                #print(x)
                #print(y)
                #print(x.shape, y.shape)
                self.current_spectrum = ax.plot(x, y, color = "green")[0]
            else:
                self.current_spectrum.set_data(x, y)
            self.current_spectrum.set_label("Current Run: %05d" % self.runidplot)

        ax.set_xlabel(r"$\lambda$ [$\AA$]")
        ax.set_ylabel(r"$F_{\lambda}$")
        ax.set_xlim([2500, 1e4])
        ax.relim()
        ax.autoscale_view(True,True,True)
        ax.legend(prop = {"size": "small"})
        self.spectrum_figure.figure.canvas.draw()


    def plot_observation(self):

        print("plotting observations")

        ax = self.spectrum_figure.figure.gca()
        if self.observation_data is not None:
            if self.obs_spectrum is None:
                self.obs_spectrum = ax.plot(self.observation_data[:,0], self.observation_data[:,1], color = "blue")[0]
            else:
                self.obs_spectrum.set_data(self.observation_data[:,0], self.observation_data[:,1])

            self.obs_spectrum.set_label(r"%s" % self.obs_fname)
            ax.set_xlabel(r"$\lambda$ [$\AA$]")
            ax.set_ylabel(r"$F_{\lambda}$")
            ax.set_xlim([2500, 1e4])
            ax.legend(prop = {"size": "small"})
            ax.relim()
            ax.autoscale_view(True,True,True)
            self.spectrum_figure.figure.canvas.draw()

    def load_observation_file(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', './')

        try:
            f = open(fname, 'r')
        except IOError:
            print("Warning: no file selected or file loading error")
            return False

        self.obs_fname = fname
        self.observation_data = np.loadtxt(f)
        f.close()

        self.plot_observation()

    def load_abundance_file(self):

        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file',  './')

        try:
            f = open(fname, 'r')
        except IOError:
            print("Warning: no file selected or file loading error")
            return False
        lines = f.readlines()
        f.close()
        try:
            assert(lines[0].rsplit()[0] == "#Abundances")
        except AssertionError:
            print("Warning: '%s' not a Tardis Abundance file" % fname)
            return False

        try:
            self.oldrunid = int(lines[0].rsplit()[1])
            self.runid = self.oldrunid + 1
        except IndexError, ValueError:
            print("Warning: could not get run-ID from header line")
            return False

        try:
            assert(lines[1].rsplit()[0] == "#risetime")
            assert(lines[2].rsplit()[0] == "#vmin")
            nstart = 3
        except AssertionError:
            try:
                assert(lines[1].rsplit()[0] == "#vmin")
                nstart = 2
            except AssertionError:
                print("Warning: '%s' not a Tardis Abundance file" % fname)
                return False

        if nstart == 3:
            self.risetime = float(lines[1].rsplit()[1])
        else:
            self.risetime = 0


        self.risetime_entry.setText(str(self.risetime))

        self.oldrunid_entry.setText(str(self.oldrunid))
        self.runid_entry.setText(str(self.runid))

        nrows = len(lines[nstart:])
        currentrowcount = self.table.rowCount()
        currentcolumncount = self.table.columnCount()

        if nrows < currentrowcount:
            for i in xrange(currentrowcount - nrows):
                self.table.removeRow(0)
        elif nrows > currentrowcount:
            for i in xrange(nrows - currentrowcount):
                self.table.insertRow(0)

        for i in xrange(nrows):
            tmp = map(float, lines[i+nstart].rsplit())
            if ((currentcolumncount - len(tmp)) == 3):
                tmp = tmp[:4] + [0,-1] + tmp[4:]
            try:
                assert(len(tmp) == currentcolumncount-1)
            except AssertionError:
                print("Warning: unexpected number of data columns")
                return False
            for j, val in enumerate(tmp):
                item = QtGui.QTableWidgetItem("%.4e" % val)
                self.table.setItem(i, j+1, item)
            item = QtGui.QTableWidgetItem("%d" % 1)
            self.table.setItem(i, 0, item)
        return True

    def save_input_files(self):

        thread = inputwriterthread(self)
        thread.starttrigger.connect(self.savingfiles_started)
        thread.endtrigger.connect(self.savingfiles_ended)
        thread.start()

    def save_abundance_file(self):

        currentrowcount = self.table.rowCount()
        currentcolumncount = self.table.columnCount()

        data = np.zeros((currentrowcount, currentcolumncount-1))
        active = np.zeros(currentrowcount, dtype = np.int)
        incomplete = False

        for i in xrange(currentrowcount):
            for j in xrange(currentcolumncount):
                val = self.table.item(i, j)
                try:
                    tmp = val.text()
                except AttributeError:
                    print("Warning: table element %d,%d not set" % (i, j))
                    incomplete = True
                    continue
                try:
                    tmpval = float(tmp)
                except ValueError:
                    print("Warning: table element %d,%d not a valid float" % (i, j))
                    incomplete = True
                    break
                if j == 0:
                    active[i] = tmpval
                else:
                    # data doesn't include active column
                    data[i,j-1] = tmpval
        if incomplete:
            print("Incomplete abundance data")
            return False
        else:
            print("Abundance data complete")
            if self.consistency_check(data):

                found_one = False
                found_one_after_zero = False

                #time check
                time = data [:,2]
                dvt=time[1:]-time[:-1]
                indices_t=np.argwhere(np.array(dvt) == 0).reshape(-1)


                for i, act in enumerate(active):

                    if act == 1:
                        found_one = True

                    else:
                        indices = np.argwhere(np.array(active[i:]) == 1).reshape(-1)
                        for j in indices:
                            found_one_after_zero = True

                    if found_one:

                        self.save_tardis_files(data[i:,:], len(active) - i)
                        found_one = False

                    elif found_one_after_zero:
                        self.save_tardis_files(data[i:,:], len(active) - i)
                        found_one_after_zero = False

                #self.save_tardis_files(data)
                self.raw_abund_data = data
                return True
            else:
                return False


    def save_tardis_files(self, data, nepoch):

        try:
            self.read_risetime()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine risetime")
            return False

        try:
            self.read_runid()
        except Exception:
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(ex_type)

            print("Warning: could not determine runids")
            return False

        #print("Starting sleep")
        #time.sleep(20)
        #print("Sleeping stopped")

        # write raw abundance file
        fname = "abundances_raw_%05d_%d.txt" % (self.runid , nepoch)

        if os.path.exists(fname):
            print("Warning: file '%s' already exists" % fname)
            return False

        f = open(fname, "w")
        f.write("#Abundances %05d\n" % self.runid)
        f.write("#risetime %.5e\n" % self.risetime)
        f.write("#vmin  vmax  t  logL/Lsun  lam min  lam max  Z = 1 - %d\n" % Zmax)
        np.savetxt(f, data)
        f.close()


        reduced_data = np.copy(data)
        #delete lam min 
        reduced_data = np.delete(reduced_data, 4, 1)
        #delete lam max
        reduced_data = np.delete(reduced_data, 4, 1)

        # write density file
        runid=self.runid
        t=reduced_data[0,2] + self.risetime
        vmin=reduced_data[0,0]
        vmax=reduced_data[-1,1]
        velocities=np.append(vmin,reduced_data[:,1])
        cr.table_densities(0.000231481,t,vmin,vmax,Nshellsfinal,runid,nepoch)

        # write abundance file
        cr.mix_abunds(velocities, 0.000231481,t,vmin,vmax,Nshellsfinal,reduced_data,runid,nepoch)

        # write Tardis yaml file
        fname = "tardis_%05d_%d.yml" % (self.runid , nepoch)
        deffname = "tardis_default.yml"

        # get default config
        try:
            tardis_default_config = yaml.safe_load(open(deffname))
        except IOError:
            print("Warning; no default parameter file ('%s') for tardis found" % deffname)
            return False

        # adjust model-dependent parameters
        tardis_default_config["supernova"]["luminosity_requested"] = "%.3f log_lsun" % data[0, 3]
        tardis_default_config["supernova"]["time_explosion"] = "%.3f day" % (data[0, 2] + self.risetime)
        tardis_default_config["model"]["structure"]["filename"] = "densities_%05d_%d.dat" % (self.runid,nepoch)
        tardis_default_config["model"]["structure"]["v_inner_boundary"] = "%.3f km/s" % data[0,0]
        tardis_default_config["model"]["structure"]["v_outer_boundary"] = "%.3f km/s" % data[-1,1]
        tardis_default_config["model"]["abundances"]["filename"] = "abundances_%05d_%d.dat" % (self.runid,nepoch)
        if data[0,4] > 0 and data[0,5] > 0:
            tardis_default_config["supernova"]["luminosity_wavelength_start"] = "%.3f angstrom" % data[0,4]
            tardis_default_config["supernova"]["luminosity_wavelength_end"] = "%.3f angstrom" % data[0,5]

        # write model-dependent Tardis config
        if os.path.exists(fname):
            print("Warning: file '%s' already exists" % fname)
            return False

        yaml.safe_dump(tardis_default_config, open(fname, "w"), default_flow_style=False)

    def consistency_check(self, data):

        #velocities check
        vmin = data[:,0]
        vmax = data[:,1]

        dv = vmax[:-1] - vmin[1:]
        indices = np.argwhere(np.fabs(dv) > 1e-3).reshape(-1)
        inconsistent = False
        for i in indices:
            print("Warning: lower velocity in shell %d not consistent with upper shell velocity of previous shell" % (i+1))
            inconsistent = True

        #abundances check (add or subtract from all elements)
        Xtot = data[:,6:].sum(axis=1)
        print Xtot

        #if np.fabs(Xtot.mean() - 1) > 1e-3:
        #    for i in xrange(data.shape[0]):
        #        data[i,4:] = data[i,4:] / Xtot[i]

        #    for i in xrange(data[:,4:].shape[0]):
        #        for j in xrange(data[:,4:].shape[1]):
        #            item = QtGui.QTableWidgetItem("%.4e" % data[i,j+4])
        #            self.table.setItem(i, j+4, item)

        #abundances check (add or subtract from oxygen)
        if np.fabs(Xtot.mean() - 1) > 1e-3:
            for i in xrange(data.shape[0]):
                add= 1-Xtot[i]
                #print data[i,13]
                data[i,13]=data[i,13]+add
                print add
                #print data[i,13]

            for i in xrange(data[:,6:].shape[0]):
                if data[i,13] > 0:
                    print "this is >0 %.3e " %data[i,13]
                    item = QtGui.QTableWidgetItem("%.4e" % data[i,13])
                    self.table.setItem(i, 14, item)
                else:
                    print ("Oxygen abundance in shell '%.3e' is negative" % i)

        #time check
        t=data[:,2]
        dvt=t[1:]-t[:-1]
        indices_t=np.argwhere(np.array(dvt)>0).reshape(-1)
        for i in indices_t:
            print ("Warning: times are not increasing inwards")
            inconsistent=True

        if inconsistent:
            return False
        else:
            return True

    def runidconvergence_entry_changed(self, text):

        self.runidconvergencetext = text

    def runidabundances_entry_changed (self,text):

        self.runidabundancestext = text

    def nepochabundances_entry_changed (self,text):

        self.nepochabundancestext = text

    def nepochlines_entry_changed (self,text):

        self.nepochlinestext = text

    def nepochplot_entry_changed (self,text):

        self.nepochplottext = text

    def runidtrads_ws_entry_changed(self,text):

        self.runidtrads_wstext = text

    def runidlines_entry_changed(self,text):

        self.runidlinestext = text

    def lamin_entry_changed(self,text):

        self.lamintext = text

    def lamax_entry_changed(self,text):

        self.lamaxtext = text

    def reddening_entry_changed(self, text):

        self.reddeningtext = text
    
    def window_entry_changed(self, text):

        self.windowtext = text

    def distance_entry_changed(self, text):

        self.distance_modulus = text

    def addshell_entry_changed(self, text):

        self.addshell_index = text

    def removeshell_entry_changed(self, text):

        self.removeshell_index = text

    def runidplot_entry_changed(self, text):

        self.runidplottext = text

    def oldrunidplot_entry_changed(self, text):

        self.oldrunidplottext = text

    def runid_entry_changed(self, text):

        self.runidtext = text

    def risetime_entry_changed(self, text):

        self.risetimetext = text

    def oldrunid_entry_changed(self, text):

        self.oldrunidtext = text

    def runidion_entry_changed(self, text):

        self.runidiontext = text

    def ion_entry_changed(self, text):

        self.iontext = text
    
    def on_addshell_clicked(self):

        currentrowcount = self.table.rowCount()
        try:
            index = int(self.addshell_index)
        except ValueError:
            print("Warning: no integer in addshell entry field - setting to 0")
            index = 0

        if index > currentrowcount:
            print("Warning: requested shell index larger than current number of shells - setting to maximum")
            index = currentrowcount

        if index < 02046:
            print("Warning: negative shell index - setting to 0")
            index = 0

        self.addshell_entry.setText(str(index))
        self.table.insertRow(index)

    def on_removeshell_clicked(self):

        currentrowcount = self.table.rowCount()
        try:
            index = int(self.removeshell_index)
        except ValueError:
            print("Warning: no integer in removeshell entry field - setting to 0")
            index = 0

        if index > currentrowcount:
            print("Warning: requested shell index larger than current number of shells - setting to maximum")
            index = currentrowcount

        if index < 0:
            print("Warning: negative shell index - setting to 0")
            index = 0

        self.removeshell_entry.setText(str(index))
        self.table.removeRow(index)


    def on_appendshell_clicked(self):

        currentcount = self.table.rowCount()
        self.table.insertRow(currentcount)


def main():

    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

