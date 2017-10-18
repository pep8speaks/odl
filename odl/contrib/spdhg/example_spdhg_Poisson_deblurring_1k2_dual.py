# ----------------------------------------------------------------------
# Matthias J. Ehrhardt
# Cambridge Image Analysis, University of Cambridge, UK
# m.j.ehrhardt@damtp.cam.ac.uk
#
# Copyright 2017 University of Cambridge
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# ----------------------------------------------------------------------

"""An example of using the SPDHG algorithm to solve a TV deblurring problem
with Poisson noise. We exploit the smoothness of the data term to get 1/k^2
convergence on the dual part.

Reference
---------
Chambolle, A., Ehrhardt, M. J., Richtárik, P., & Schönlieb, C.-B. (2017).
Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and
Imaging Applications. Retrieved from http://arxiv.org/abs/1706.04957
"""

from __future__ import print_function
from __future__ import division
import odl
import odl.contrib.spdhg as spdhg
import odl.contrib.fom as odl_fom
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import misc as sc_misc
from skimage.io import imsave as sk_imsave

__author__ = 'Matthias J. Ehrhardt'
__copyright__ = 'Copyright 2017, University of Cambridge'

# create folder for data TO BE CHANGED!!
folder_out = '.'
filename = 'example_spdhg_Poisson_deblurring_1k2_dual'

# load image
image_raw = np.rot90(sc_misc.imread('rings.jpg').astype('float32'), 3)
image_cropped = image_raw[:, 300:300+image_raw.shape[0], :]
scale_image = 10  # 4
s_image = (int(image_cropped.shape[0] / scale_image),
           int(image_cropped.shape[1] / scale_image))
filename = '{}_{}x{}'.format(filename, s_image[0], s_image[1])

n_epochs = 300  # number of epochs
n_iter_target = 10000

subfolder = '{}_{}epochs'.format('20171002_1400', n_epochs)

folder_main = '{}/{}'.format(folder_out, filename)
spdhg.mkdir(folder_main)

folder_today = '{}/{}'.format(folder_main, subfolder)
spdhg.mkdir(folder_today)

folder_npy = '{}/npy'.format(folder_today)
spdhg.mkdir(folder_npy)

# create ground truth
image_resized = sc_misc.imresize(
        np.rot90(image_cropped, 3), s_image).astype('float32')
image_gray = np.sum(image_resized, 2)
X = odl.uniform_discr([0, 0], s_image, s_image)
groundtruth = 100 * X.element(image_gray / np.max(image_gray))
clim = [0, 100]
cmap = 'gray'
tol_norm = 1.05

# create forward operators
Dx = odl.PartialDerivative(X, 0, pad_mode='symmetric')
Dy = odl.PartialDerivative(X, 1, pad_mode='symmetric')

kernel_raw = np.float64(255 - sc_misc.imread('motionblur.png'))
s_kernel = [9] * 2
kernel_resized = np.float64(sc_misc.imresize(kernel_raw, s_kernel))
kernel = kernel_resized
kernel /= np.sum(kernel)
convolution = spdhg.Blur(X, kernel)

mask = X.zero()
b0 = int(0.6 * kernel.shape[0])
b1 = int(0.6 * kernel.shape[1])
mask.asarray()[b0:-b1, b0:-b1] = 1
sampling_points = np.where(mask.asarray() == 1)
sampling = odl.SamplingOperator(convolution.range, sampling_points)
scale = 1e+3
A = odl.BroadcastOperator(Dx, Dy, scale / clim[1] * convolution)
Y = A.range

# create data
clim_data = [0, scale]
background = 30 * Y[2].one()
data = odl.phantom.poisson_noise(A[2](groundtruth) + background, seed=1807)

# save images and data
if not spdhg.exists('{}/groundtruth.png'.format(folder_main)):
    fig1 = plt.figure(1)
    groundtruth.show('ground truth', clim=clim, cmap=cmap, fig=fig1)

    fig2 = plt.figure(2)
    data.show('data', clim=clim_data, cmap=cmap, fig=fig2)

    fig1.savefig('{}/groundtruth.png'.format(folder_main), bbox_inches='tight')
    fig2.savefig('{}/data.png'.format(folder_main), bbox_inches='tight')

    k = kernel / np.max(kernel)
    sk_imsave('{}/kernel.png'.format(folder_main), k)

# set regularisation parameter
alpha = 0.1
# set up functional f
f = odl.solvers.SeparableSum(
        alpha * spdhg.HuberL1(A[0].range, gamma=1),
        alpha * spdhg.HuberL1(A[1].range, gamma=1),
        spdhg.KullbackLeiblerSmooth(A[2].range, data, background))

# set up functional g
g = odl.solvers.IndicatorBox(X, clim[0], clim[1])

# define objective function
objFun = f * A + g

# define strong convexity constants
mu_i = [1 / fi.grad_lipschitz for fi in f]
mu_f = min(mu_i)

# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not spdhg.exists(file_target):

    # compute norm of operator
    normA = tol_norm * A.norm(estimate=True, xstart=odl.phantom.white_noise(X))

    # set step size parameters
    sigma = 1 / normA
    tau = 1 / normA

    # initialise variables
    x_opt = X.zero()
    y_opt = Y.zero()

    # define callback for visual output during iterations
    callback = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}',
                                                   step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s',
                                                step=10, cumulative=True))

    # compute a saddle point with PDHG and time the reconstruction
    odl.solvers.pdhg(x_opt, f, g, A, tau, sigma, n_iter_target, y=y_opt,
                     callback=callback)

    # compute the subgradients of the saddle point
    q_opt = -A.adjoint(y_opt)
    r_opt = A(x_opt)

    # compute the objective function value at the saddle point
    obj_opt = objFun(x_opt)

    # save saddle point
    np.save(file_target, (x_opt, y_opt, q_opt, r_opt, obj_opt, normA))

    # show saddle point and subgradients
    fig1 = plt.figure(1)
    x_opt.show('x saddle', clim=clim, cmap=cmap, fig=fig1)

    fig2 = plt.figure(2)
    y_opt[0].show('y saddle[0]', fig=fig2)

    fig3 = plt.figure(3)
    q_opt.show('q saddle', fig=fig3)

    fig4 = plt.figure(4)
    r_opt[0].show('r saddle[0]', fig=fig4)

    # save images
    fig1.savefig('{}/x_opt.png'.format(folder_main), bbox_inches='tight')
    fig2.savefig('{}/y_opt.png'.format(folder_main), bbox_inches='tight')
    fig3.savefig('{}/q_opt.png'.format(folder_main), bbox_inches='tight')
    fig4.savefig('{}/r_opt.png'.format(folder_main), bbox_inches='tight')

else:
    (x_opt, y_opt, q_opt, r_opt, obj_opt, normA) = np.load(file_target)

# set verbosity level
verbose = 1

# set norms of the primal and dual variable
norm_x_sq = 1 / 2 * odl.solvers.L2NormSquared(X)
norm_y_sq = 1 / 2 * odl.solvers.L2NormSquared(Y)

# create Bregman distances for f and g
bregman_g = spdhg.bregman(g, x_opt, q_opt)
bregman_f = odl.solvers.SeparableSum(*[spdhg.bregman(fi.convex_conj, yi, ri)
                                       for fi, yi, ri in zip(f, y_opt, r_opt)])

dist_x = norm_x_sq.translated(x_opt)
dist_y = norm_y_sq.translated(y_opt)


# define callback to store function values
class CallbackStore(odl.solvers.Callback):

    def __init__(self, iter_save_data):
        self.iter_save_data = iter_save_data
        self.iteration_count = 0
        self.ergodic_iterate_x = 0
        self.ergodic_iterate_y = 0
        self.out = []

    def __call__(self, x):
        k = self.iteration_count

        if k > 0:
            self.ergodic_iterate_x = 1 / k * (
                    (k - 1) * self.ergodic_iterate_x + x[0])
            self.ergodic_iterate_y = 1 / k * (
                    (k - 1) * self.ergodic_iterate_y + x[1])
        else:
            self.ergodic_iterate_x = X.zero()
            self.ergodic_iterate_y = Y.zero()

        if k in self.iter_save_data:
            obj = objFun(x[0])
            breg_x = bregman_g(x[0])
            breg_y = bregman_f(x[1])
            breg = breg_x + breg_y
            breg_erg_x = bregman_g(self.ergodic_iterate_x)
            breg_erg_y = bregman_f(self.ergodic_iterate_y)
            breg_erg = breg_erg_x + breg_erg_y
            psnr = odl_fom.psnr(x[0], groundtruth)
            psnr_opt = odl_fom.psnr(x[0], x_opt)

            self.out.append({'obj': obj, 'breg': breg, 'breg_x': breg_x,
                             'breg_y': breg_y, 'breg_erg': breg_erg,
                             'breg_erg_x': breg_erg_x,
                             'breg_erg_y': breg_erg_y,
                             'dist_x': dist_x(x[0]), 'dist_y': dist_y(x[1]),
                             'psnr': psnr, 'psnr_opt': psnr_opt, 'iter': k})

        self.iteration_count += 1


# number of subsets for each algorithm
n_subsets = dict()
n_subsets['pdhg'] = 1
n_subsets['da_pdhg'] = 1
n_subsets['spdhg_uni3'] = 3
n_subsets['spdhg_importance3'] = 3
n_subsets['da_spdhg_uni3'] = 3
n_subsets['da_spdhg_importance3'] = 3

# number of iterations for each algorithm
n_iter = {}
iter_save_data = {}
for alg in n_subsets.keys():
    n_iter[alg] = n_epochs * n_subsets[alg]
    iter_save_data[alg] = range(0, n_iter[alg] + 1, n_subsets[alg])

# %%
# run algorithms
for alg in ['pdhg', 'da_pdhg', 'spdhg_uni3', 'spdhg_importance3',
            'da_spdhg_uni3', 'da_spdhg_importance3']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob_subset = None
    prob = None
    extra = None
    sigma = None
    sigma_tilde = None
    tau = None
    theta = None

    # set random seed so that results are reproducable
    np.random.seed(1807)

    # create lists for subset division
    n = n_subsets[alg]
    (subset2ind, ind2subset) = spdhg.divide_1Darray_equally(range(Y.size), n)

    file_normA = '{}/norms_{}subsets.npy'.format(folder_main, n)
    if not spdhg.exists(file_normA):
        if n == 1:
            normA = [tol_norm * A.norm(estimate=True,
                                       xstart=odl.phantom.white_noise(X))]
        elif n == 3:
            normA = [2, 2,
                     tol_norm * A[2].norm(estimate=True,
                                          xstart=odl.phantom.white_noise(X))]
        np.save(file_normA, normA)

    else:
        normA = np.load(file_normA)

    # choose parameters for algorithm
    if alg == 'pdhg':
        prob_subset = [1] * n
        prob = [1] * Y.size
        sigma = [1 / normA[0]] * Y.size
        tau = 1 / normA[0]

    elif alg == 'da_pdhg':
        prob_subset = [1] * n
        prob = [1] * Y.size
        extra = [1] * Y.size
        tau = 1 / normA[0]
        mu = [mu_f] * Y.size
        sigma_tilde = mu_f / normA[0]

    elif alg in ['spdhg_uni3']:
        prob = [1 / n] * n
        prob_subset = prob
        sigma = [1 / normAi for normAi in normA]
        tau = 1 / (n * max(normA))

    elif alg in ['spdhg_importance3']:
        prob = [normAi / sum(normA) for normAi in normA]
        prob_subset = prob
        sigma = [1 / normAi for normAi in normA]
        tau = 1 / sum(normA)

    elif alg in ['da_spdhg_uni3']:
        prob = [1 / n] * n
        prob_subset = prob
        extra = [1 / p for p in prob]
        tau = 1 / (n * max(normA))
        mu = mu_i
        sigma_tilde = min([mu * p**2 / (tau * normAi**2 + 2 * mu * p * (1 - p))
                           for p, mu, normAi in zip(prob, mu_i, normA)])

    elif alg in ['da_spdhg_importance3']:
        prob = [normAi / sum(normA) for normAi in normA]
        prob_subset = prob
        extra = [1 / p for p in prob]
        tau = 1 / sum(normA)
        mu = mu_i
        sigma_tilde = min([mu * p**2 / (tau * normAi**2 + 2 * mu * p * (1 - p))
                           for p, mu, normAi in zip(prob, mu_i, normA)])

    else:
        raise NameError('Parameters not defined')

    # function that selects the indices every iteration
    def fun_select(k):
        return subset2ind[int(np.random.choice(n, 1, p=prob_subset))]

    # initialise variables
    x = X.zero()
    y = Y.zero()

    # output function to be used within the iterations
    callback = (odl.solvers.CallbackPrintIteration(fmt='iter:{:4d}',
                                                   step=verbose * n,
                                                   end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time/iter: {:5.2f} s',
                                                step=verbose * n, end=', ') &
                odl.solvers.CallbackPrintTiming(fmt='time: {:5.2f} s',
                                                cumulative=True,
                                                step=verbose * n) &
                CallbackStore(iter_save_data[alg]))

    callback([x, y])

    if alg[:4] == 'pdhg' or alg[:5] == 'spdhg':
        spdhg.spdhg(x, f, g, A, tau, sigma, n_iter[alg], prob, fun_select, y=y,
                    callback=callback)

    elif alg[:7] == 'da_pdhg' or alg[:8] == 'da_spdhg':
        spdhg.da_spdhg(x, f, g, A, tau, sigma_tilde, n_iter[alg], extra, prob,
                       mu, fun_select, y=y, callback=callback)

    else:
        raise NameError('Algorithm not defined')

    output = callback.callbacks[1].out

    np.save('{}/{}_{}_output'.format(folder_npy, filename, alg),
            (iter_save_data[alg], n_iter[alg], x, output, n_subsets[alg]))

# %%
algorithms = ['pdhg', 'da_pdhg', 'spdhg_uni3', 'spdhg_importance3',
              'da_spdhg_uni3', 'da_spdhg_importance3']
iter_save_data_v = {}
n_iter_v = {}
image_v = {}
output_v = {}
n_subsets_v = {}
for alg in algorithms:
    (iter_save_data_v[alg], n_iter_v[alg], image_v[alg], output_v[alg],
     n_subsets_v[alg]) = np.load('{}/{}_{}_output.npy'.format(folder_npy,
                                 filename, alg))

epochs_save_data = {}
for alg in algorithms:
    n = np.float(n_subsets_v[alg])
    epochs_save_data[alg] = np.array(iter_save_data_v[alg]) / n

output_resorted = {}
for alg in algorithms:
    print('==== ' + alg)
    output_resorted[alg] = {}
    K = len(iter_save_data_v[alg])

    for meas in output_v[alg][0].keys():  # quality measures
        print('    ==== ' + meas)
        output_resorted[alg][meas] = np.nan * np.ones(K)

        for k in range(K):  # iterations
            output_resorted[alg][meas][k] = output_v[alg][k][meas]

    meas = 'obj_rel'
    print('    ==== ' + meas)
    output_resorted[alg][meas] = np.nan * np.ones(K)

    for k in range(K):  # iterations
        output_resorted[alg][meas][k] = ((output_v[alg][k]['obj'] - obj_opt) /
                                         (output_v[alg][0]['obj'] - obj_opt))

for alg in algorithms:  # algorithms
    for meas in output_resorted[alg].keys():  # quality measures
        for k in range(K):  # iterations
            if output_resorted[alg][meas][k] <= 0:
                output_resorted[alg][meas][k] = np.nan

# %%
fig = plt.figure(1)
plt.clf()

# set latex options
matplotlib.rc('text', usetex=False)

for alg in algorithms:
    print('==== ' + alg)
    image_v[alg].show(alg, clim=clim, cmap=cmap, fig=fig)
    fig.savefig('{}/{}.png'.format(folder_today, alg), bbox_inches='tight')

markers = plt.Line2D.filled_markers

all_plots = ['dist_x', 'dist_y', 'obj', 'obj_rel', 'breg_erg', 'breg_erg_x',
             'breg_erg_y', 'breg', 'breg_y', 'breg_x', 'psnr', 'psnr_opt']
logy_plot = ['obj', 'obj_rel', 'dist_x', 'dist_y', 'breg', 'breg_y', 'breg_x',
             'breg_erg', 'breg_erg_x', 'breg_erg_y']

for plotx in ['normalx', 'logx']:
    for meas in all_plots:
        print('============ ' + plotx + ' === ' + meas + ' ============')
        fig = plt.figure(1)
        plt.clf()

        if plotx == 'normalx':
            if meas in logy_plot:
                for alg in algorithms:
                    x = epochs_save_data[alg]
                    y = output_resorted[alg][meas]
                    plt.semilogy(x, y, linewidth=3, label=alg)
            else:
                for jj, alg in enumerate(algorithms):
                    x = epochs_save_data[alg]
                    y = output_resorted[alg][meas]
                    plt.plot(x, y, linewidth=3, marker=markers[jj],
                             markersize=7, markevery=.1, label=alg)

        elif plotx == 'logx':
            if meas in logy_plot:
                for alg in algorithms:
                    x = epochs_save_data[alg][1:]
                    y = output_resorted[alg][meas][1:]
                    plt.loglog(x, y, linewidth=3, label=alg)
            else:
                for jj, alg in enumerate(algorithms):
                    x = epochs_save_data[alg][1:]
                    y = output_resorted[alg][meas][1:]
                    plt.semilogx(x, y, linewidth=3, marker=markers[jj],
                                 markersize=7, markevery=.1, label=alg)

        plt.title('{} v iteration'.format(meas))
        h = plt.gca()
        h.set_xlabel('epochs')
        plt.legend(loc='best')

        fig.savefig('{}/{}_{}_{}.png'.format(folder_today, filename,
                    plotx, meas), bbox_inches='tight')

# %%
# set line width and style
lwidth = 2
lwidth_help = 2
lstyle = '-'
lstyle_help = '--'

# set colors using colorbrewer
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 6)
colors = bmap.mpl_colors

# set latex options
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

# set font
fsize = 15
font = {'family': 'serif', 'size': fsize}
matplotlib.rc('font', **font)
matplotlib.rc('axes', labelsize=fsize)  # fontsize of x and y labels
matplotlib.rc('xtick', labelsize=fsize)  # fontsize of xtick labels
matplotlib.rc('ytick', labelsize=fsize)  # fontsize of ytick labels
matplotlib.rc('legend', fontsize=fsize)  # legend fontsize

# markers
marker = ('o', 'v', 's', 'p', 'd')  # set markers
mevery = [(i / 30., .1) for i in range(20)]  # how many markers to draw
msize = 9  # marker size

fig = []

# draw first figure
fig.append(plt.figure(1))
plt.clf()

meas = 'dist_y'
label = ['PDHG', 'SPDHG (importance)', 'DA-PDHG', 'DA-SPDHG (importance)']
for k, alg in enumerate(['pdhg', 'spdhg_importance3', 'da_pdhg',
                         'da_spdhg_importance3']):
    x = epochs_save_data[alg][1:]
    y = output_resorted[alg][meas][1:]
    plt.loglog(x, y, color=colors[k], linestyle=lstyle, linewidth=lwidth,
               marker=marker[k], markersize=msize, markevery=mevery[k],
               label=label[k])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('dual distance')
plt.gca().yaxis.set_ticks(np.logspace(1, 5, 3))

# set x and y limits
plt.ylim((1e-0, 1e+6))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# ### next figure
fig.append(plt.figure(2))
plt.clf()

max_iter = 300
min_y = 1e-5
meas = 'obj_rel'
names = ['PDHG', 'SPDHG (importance)', 'DA-PDHG',
         'DA-SPDHG (importance)']
for k, alg in enumerate(['pdhg', 'spdhg_importance3', 'da_pdhg',
                         'da_spdhg_importance3']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    i = np.less(x, max_iter+1) & np.greater(x, 0) & np.greater(y, min_y)
    plt.loglog(x[i], y[i], color=colors[k], linestyle=lstyle, linewidth=lwidth,
               marker=marker[k], markersize=msize, markevery=mevery[k],
               label=label[k])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative objective')
plt.gca().yaxis.set_ticks(np.logspace(-4, 0, 3))

# set x and y limits
plt.ylim((min_y, 2e+0))
plt.xlim((1, max_iter))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# ### next figure
fig.append(plt.figure(3))
plt.clf()

meas = 'psnr'
names = ['PDHG', 'SPDHG (importance)', 'DA-PDHG', 'DA-SPDHG (importance)']
for k, alg in enumerate(['pdhg', 'spdhg_importance3', 'da_pdhg',
                         'da_spdhg_importance3']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    plt.plot(x, y, color=colors[k], linestyle=lstyle, linewidth=lwidth,
             marker=marker[k], markersize=msize, markevery=mevery[k],
             label=label[k])

plt.axhline(y=output_resorted['pdhg'][meas][-1], color='gray',
            linestyle='--', linewidth=lwidth_help)
plt.axvline(x=30, color='gray', linestyle='--', linewidth=lwidth_help)

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('PSNR')
plt.gca().yaxis.set_ticks(np.linspace(10, 30, 3))

# set x and y limits
plt.ylim((5, 30))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# ### next figure
fig.append(plt.figure(4))
plt.clf()

meas = 'dist_x'
names = ['PDHG', 'SPDHG (importance)', 'DA-PDHG', 'DA-SPDHG (importance)']
for k, alg in enumerate(['pdhg', 'spdhg_importance3', 'da_pdhg',
                         'da_spdhg_importance3']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    plt.semilogy(x, y, color=colors[k], linestyle=lstyle, linewidth=lwidth,
                 marker=marker[k], markersize=msize, markevery=mevery[k],
                 label=label[k])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('primal distance')
plt.gca().yaxis.set_ticks(np.logspace(4, 8, 3))

# set x and y limits
plt.ylim((1e+4, 1e+8))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/{}_output{}.png'.format(folder_today, filename, i),
               bbox_inches='tight')
