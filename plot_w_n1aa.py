#%%
import glob, pandas as pd
import numpy as np, os.path as op
from matplotlib import pyplot as plt

def get_bias(path, imin=30):
    biases = []
    for i in range(10):
        files = glob.glob(op.join(path, f"*_CMBLensed_fullsky_alm_{i:03d}_*.csv"))
        rot_files = glob.glob(op.join(path, f"*_CMBLensedRot_fullsky_alm_{i:03d}_*.csv"))
        lensed = np.stack([pd.read_csv(f).values for f in files], axis=0)
        rot_lensed = np.stack([pd.read_csv(f).values for f in rot_files], axis=0) 
        ell = np.arange(lensed.shape[1])[imin:]
        clpp = np.mean(lensed, axis=0)[imin:,0]
        xclpp = np.mean(lensed, axis=0)[imin:,1]
        rdn0 = np.mean(lensed, axis=0)[imin:,2]
        
        clpp_rot = np.mean(rot_lensed, axis=0)[imin:,0]
        xclpp_rot = np.mean(rot_lensed, axis=0)[imin:,1]
        rdn0_rot = np.mean(rot_lensed, axis=0)[imin:,2]

        bias = clpp_rot - clpp - rdn0_rot + rdn0
        biases.append(bias)
    return ell, biases

def get_rdn0(path, imin=30):
    rdn0s = []
    for i in range(10):
        files = glob.glob(op.join(path, f"*_CMBLensed_fullsky_alm_{i:03d}_*.csv"))
        rot_files = glob.glob(op.join(path, f"*_CMBLensedRot_fullsky_alm_{i:03d}_*.csv"))
        lensed = np.stack([pd.read_csv(f).values for f in files], axis=0)
        rot_lensed = np.stack([pd.read_csv(f).values for f in rot_files], axis=0) 
        ell = np.arange(lensed.shape[1])[imin:]
        rdn0 = np.mean(lensed, axis=0)[imin:,2]
        rdn0_rot = np.mean(rot_lensed, axis=0)[imin:,2]
        # rdn0s.append(rdn0)
        rdn0s.append(rdn0)
    return ell, rdn0s

def load_n1aa(Acb="1e-08", expt="S4"):
    files = glob.glob(f"output/N1aa/N1aa_*_{Acb}_CMB_{expt}*.dat")
    n1aa = []
    for f in files:
        n1aa.append(np.loadtxt(f))
    n1aa = np.mean(n1aa, axis=0)[30:3001]
    return n1aa

class bin1D(object):
    def __init__(self, ls, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2.
        self.digitized = np.digitize(ls, bin_edges,right=True)
        self.bin_edges = bin_edges
    def bin(self,data1d,weights=None):
        if weights is None:
            res = np.bincount(self.digitized,(data1d).reshape(-1))[1:-1]/np.bincount(self.digitized)[1:-1]
        else:
            res = np.bincount(self.digitized,(data1d*weights).reshape(-1))[1:-1]/np.bincount(self.digitized,weights.reshape(-1))[1:-1]
        return self.centers,res

ell, biases = get_bias("./output/recon_ps_s4")
_, biases_s3 = get_bias("./output/recon_ps_s3")
_, biases_n0 = get_bias("./output/recon_ps_n0")

# load input
clpp_input = np.loadtxt("./inputPs/cosmo2017_10K_acc3_lenspotentialCls.dat")[30:3001,5]
clkk = 2*np.pi/4*clpp_input
clpp = clpp_input / ((ell*(ell+1))**2 / (2*np.pi))

biases /= clkk
biases_s3 /= clkk
biases_n0 /= clkk

n1aa_s4 = load_n1aa(expt="S4")
n1aa_s3 = load_n1aa(expt="S3")

# %%
# bin results
bin_edge = np.arange(30, 3000, 150)
binner = bin1D(ell, bin_edge)

# bin each spectra
biases_bin = []
biases_bin_s3 = []
biases_bin_n0 = []

for bias, bias_s3, bias_n0 in zip(biases, biases_s3, biases_n0):
    ell_bin, bias_bin = binner.bin(bias)
    ell_bin, bias_bin_s3 = binner.bin(bias_s3)
    ell_bin, bias_bin_n0 = binner.bin(bias_n0)
    biases_bin.append(bias_bin)
    biases_bin_s3.append(bias_bin_s3)
    biases_bin_n0.append(bias_bin_n0)

biases_bin = np.array(biases_bin)
biases_bin_s3 = np.array(biases_bin_s3)
biases_bin_n0 = np.array(biases_bin_n0)

# %%
bias_mean = np.mean(biases_bin, axis=0)
bias_err  = np.std(biases_bin, axis=0)

bias_mean_s3 = np.mean(biases_bin_s3, axis=0)
bias_err_s3  = np.std(biases_bin_s3, axis=0)

bias_mean_n0 = np.mean(biases_bin_n0, axis=0)
bias_err_n0  = np.std(biases_bin_n0, axis=0)

_, n1aa_s4_bin = binner.bin(n1aa_s4 / clpp)
_, n1aa_s3_bin = binner.bin(n1aa_s3 / clpp)
#%%
plt.rcParams["font.size"] = 14
# plt.rcParams["font.size"] = 14
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["axes.labelsize"] = 14
# plt.rcParams["text.usetex"] = True
# fig = plt.figure(figsize=(6,4), dpi=140)
fig = plt.figure(dpi=140)
# ax = fig.add_subplot(111)
plt.errorbar(ell_bin, bias_mean, yerr=bias_err, c="r", label=r"Bias S4")
plt.errorbar(ell_bin, bias_mean_s3, yerr=bias_err_s3, c="b", label=r"Bias S3")
# plt.errorbar(ell_bin, bias_mean_n0, yerr=bias_err_n0, label=r"Noiseless")
plt.plot(ell_bin, n1aa_s4_bin, "r--", label=r"$N_L^{(1)}$ S4")
plt.plot(ell_bin, n1aa_s3_bin, "b--", label=r"$N_L^{(1)}$ S3")
# get twin x-axis
# ax2 = ax.twinx()
# ax2 = ax
# c = 8e-4
# ax2.plot(ell, -rdn0_rot/clkk*c, label="Ref S4", c='r')
# ax2.plot(ell, -rdn0_rot_s3/clkk*c, label="Ref S3", c='r')
plt.legend(loc="best")
plt.xlabel("$L$")
plt.ylabel('$(\Delta \hat{{C}}^{\mathrm{\phi \phi}}_{L})_{\mathrm{rot}}/C^{\mathrm{\phi \phi}}_{L}$')
plt.title(r"$A_{\rm CB}=10^{-7}$")
plt.savefig("bias_expt_w_n1.pdf", bbox_inches="tight")

# %%
# compare with rdn0
ell, rdn0s = get_rdn0("./output/recon_ps_s4")
_, rdn0s_s3 = get_rdn0("./output/recon_ps_s3")
_, rdn0s_n0 = get_rdn0("./output/recon_ps_n0")

# fractional
rdn0s /= clkk
rdn0s_s3 /= clkk
rdn0s_n0 /= clkk

# bin results
rdn0s_bin = np.array([binner.bin(rdn0)[1] for rdn0 in rdn0s])
rdn0s_bin_s3 = np.array([binner.bin(rdn0)[1] for rdn0 in rdn0s_s3])
rdn0s_bin_n0 = np.array([binner.bin(rdn0)[1] for rdn0 in rdn0s_n0])

# calculate mean and std
rdn0_mean = np.mean(rdn0s_bin, axis=0)
rdn0_err  = np.std(rdn0s_bin, axis=0)
rdn0_mean_s3 = np.mean(rdn0s_bin_s3, axis=0)
rdn0_err_s3  = np.std(rdn0s_bin_s3, axis=0)
rdn0_mean_n0 = np.mean(rdn0s_bin_n0, axis=0)
rdn0_err_n0  = np.std(rdn0s_bin_n0, axis=0)
# %% 
# plot results
fig = plt.figure(figsize=(6,4), dpi=140)
ax = fig.add_subplot(111)
# ax.errorbar(ell_bin, bias_mean, yerr=bias_err, label=r"CMB S4, $A_{\rm CB}=10^{-7}$")
# ax.errorbar(ell_bin, bias_mean_s3, yerr=bias_err_s3, label=r"CMB S3, $A_{\rm CB}=10^{-7}$")
ax.plot(ell_bin, bias_mean, label=r"bias (CMB S4)")
ax.plot(ell_bin, bias_mean_s3, label=r"bias (CMB S3)")
ax.plot(ell_bin, bias_mean_n0, label=r"bias (Noiseless)")
ax.set_xlabel("$L$")
ax.set_ylabel('$(\Delta \hat{{C}}^{\mathrm{\phi \phi}}_{L})_{\mathrm{rot}}/C^{\mathrm{\phi \phi}}_{L}$')
ax.legend(loc="center left")
# get twin x-axis
ax2 = ax.twinx()
# ax2 = ax
# c = 5.5e-4
c = 1
ax2.plot(ell_bin, -rdn0_mean*c, label="N0 (CMB S4)", ls="--")
ax2.plot(ell_bin, -rdn0_mean_s3*c, label="N0 (CMB S3)", ls="--")
ax2.plot(ell_bin, -rdn0_mean_n0*c, label="N0 (Noiseless)", ls="--")
ax2.set_ylabel(r"$-N^{(0)}_L/C^{\mathrm{\phi \phi}}_{L}$")
ax2.legend(loc="lower left")
ax2.set_title(r"$A_{\rm CB}=10^{-7}$")
# plt.savefig("bias_n0.pdf", bbox_inches="tight")

# ax.set_yticks(np.linspace(0, ax.get_ybound()[0], 10))
# ax2.set_yticks(np.linspace(0, ax2.get_ybound()[0], 10))
# plt.savefig("bias_expt.pdf", bbox_inches="tight")

# %%
# plot results with different ACB
_, biases_n6 = get_bias("./output/recon_ps_n6")
_, biases_n7 = get_bias("./output/recon_ps_n7")
_, biases_n8 = get_bias("./output/recon_ps_n8")
biases_n6 /= clkk
biases_n7 /= clkk
biases_n8 /= clkk

bias_n6_bin = np.array([binner.bin(bias)[1] for bias in biases_n6])
bias_n7_bin = np.array([binner.bin(bias)[1] for bias in biases_n7])
bias_n8_bin = np.array([binner.bin(bias)[1] for bias in biases_n8])

# calculate mean and std
bias_n6_mean = np.mean(bias_n6_bin, axis=0)
bias_n6_err = np.std(bias_n6_bin, axis=0)
bias_n7_mean = np.mean(bias_n7_bin, axis=0)
bias_n7_err = np.std(bias_n7_bin, axis=0)
bias_n8_mean = np.mean(bias_n8_bin, axis=0)
bias_n8_err = np.std(bias_n8_bin, axis=0)

# n1
n1aa_n6 = load_n1aa(Acb="1e-06")
n1aa_n7 = load_n1aa(Acb="1e-07")
n1aa_n8 = load_n1aa(Acb="1e-08")

_, n1aa_n6_bin = binner.bin(n1aa_n6 / clpp)
_, n1aa_n7_bin = binner.bin(n1aa_n7 / clpp)
_, n1aa_n8_bin = binner.bin(n1aa_n8 / clpp)
# %%

fig = plt.figure(dpi=140)
ax = fig.add_subplot(111)
ax.errorbar(ell_bin, bias_n6_mean/10, yerr=bias_n6_err/10, fmt="r.",label=r'$(A_{\mathrm{CB}}=10^{-6})/10$')
ax.errorbar(ell_bin+30, bias_n7_mean, yerr=bias_n7_err, fmt="g.", label=r'$A_{\mathrm{CB}}=10^{-7}$')
ax.errorbar(ell_bin+60, bias_n8_mean*10, yerr=bias_n8_err*10, fmt="b.", label=r'$(A_{\mathrm{CB}}=10^{-8})\times 10$')
ax.plot(ell_bin, n1aa_n6_bin/10, "r--", label=r'$N^{(1)}_{L}$: $(A_{\mathrm{CB}}=10^{-6})/10$', alpha=0.5)
ax.plot(ell_bin, n1aa_n7_bin, "g--", label=r'$N^{(1)}_{L}$: $A_{\mathrm{CB}}=10^{-7}$', alpha=0.5)
ax.plot(ell_bin, n1aa_n8_bin*10, "b--", label=r'$N^{(1)}_{L}$: $(A_{\mathrm{CB}}=10^{-8})\times 10$', alpha=0.5)
ax.set_xlabel("$L$")
ax.set_ylabel('$(\Delta \hat{{C}}^{\mathrm{\phi \phi}}_{L})_{\mathrm{rot}}/C^{\mathrm{\phi \phi}}_{L}$')
ax.legend()
plt.savefig("bias_scaling_w_n1aa.pdf", bbox_inches='tight')

# %%
# calculate total snr for cmb s4
# _, biases_n7 = get_bias("./output/recon_ps_s4")
# _, rdn0s = get_rdn0("./output/recon_ps_s4")
_, biases_n7 = get_bias("./output/recon_ps_n7")
_, rdn0s = get_rdn0("./output/recon_ps_n7")

bias_n7 = np.mean(biases_n7, axis=0)
rdn0_n7 = np.mean(rdn0s, axis=0)
n0_n7 = np.loadtxt("n0.txt")

fsky = 0.4
# snr = np.sum(bias_n7**2 / (clkk + rdn0s)**2 * ell * fsky)**0.5
# snr = np.sum(clkk**2 / (clkk + n0_n7[ell])**2 * ell * fsky)**0.5
snr = np.sum(bias_n7**2 / (clkk + n0_n7[ell])**2 * ell * fsky)**0.5

print("snr = ", snr)
# %%
