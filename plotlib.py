import matplotlib as mpl
from cycler import cycler

# common style for all plots
mpl.rcParams['figure.dpi']  = 180
mpl.rcParams['font.size']   = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = False 
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.rcParams['font.family'] = 'serif'

def setup_axis(ax, xlabel=None, ylabel=None, xscale=None, yscale=None, 
               fs=18, lbl_fs=None, title=None):
    if lbl_fs is None: lbl_fs = fs
    if xlabel: ax.set_xlabel(xlabel, fontsize=lbl_fs)
    if ylabel: ax.set_ylabel(ylabel, fontsize=lbl_fs)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if title:  ax.set_title(title, fontsize=lbl_fs)
    return ax

def texify(text):
    text = text.replace(" ", "~")
    return r"${\rm %s}$" % text
