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

"""An example of using the SPDHG algorithm to solve a PET reconstruction
problem with a strongly convex total variation prior. We exploit the smoothness
of the data term and the strong convexity of the prior to obtain a linearly
convergent algorithm.

Note that this example uses the astra toolbox.

Reference
---------
Chambolle, A., Ehrhardt, M. J., Richtárik, P., & Schönlieb, C.-B. (2017).
Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and
Imaging Applications. Retrieved from http://arxiv.org/abs/1706.04957
"""

from __future__ import print_function
from __future__ import division
import odl.contrib.spdhg as spdhg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter as sc_gaussian
from scipy import misc as sc_misc
from skimage.io import imsave as sk_imsave
import odl
import brewer2mpl

__author__ = "Matthias J. Ehrhardt"
__copyright__ = "Copyright 2017, University of Cambridge"

# create folder for data TO BE CHANGED!!
folder_out = '.'
filename = "example_spdhg_PET_linear_rate"

# set number of epochs
n_epochs = 30
#n_epochs = 10
n_iter_target = 2000

# subfolder will contain date and number of epochs
subfolder = '{}_{}epochs'.format('20171002_1400', n_epochs)

# set problem size
n_voxel_x = 200

# change filename with problem size
filename = '{}_{}x{}'.format(filename, n_voxel_x, n_voxel_x)

# create output folders
folder_main = '{}/{}'.format(folder_out, filename)
spdhg.mkdir(folder_main)

folder_today = '{}/{}'.format(folder_main, subfolder)
spdhg.mkdir(folder_today)

folder_npy = '{}/npy'.format(folder_today)
spdhg.mkdir(folder_npy)

# create geometry of operator
X = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                      shape=[n_voxel_x, n_voxel_x], dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(X, num_angles=200, det_shape=200)

G = odl.BroadcastOperator(*[odl.tomo.RayTransform(X, gi, impl='astra_cpu')
                            for gi in geometry])

# create ground truth
Y = G.range
image_raw = np.rot90(sc_misc.imread('PET_phantom.png'), 3)
image_resized = sc_misc.imresize(image_raw, X.shape).astype('float32')
groundtruth = X.element(image_resized / np.max(image_resized))
clim = [0, 1]
tol_norm = 1.05

# save images and data
file_data = '{}/data.npy'.format(folder_main)
if not spdhg.exists(file_data):
    sinogram = G(groundtruth)

    factors = -G(0.005 / X.cell_sides[0] * groundtruth.ufuncs.greater(0))
    factors.ufuncs.exp(out=factors)

    counts_observed = (factors * sinogram).ufuncs.sum()
    counts_desired = 3e+6
    counts_background = 2e+6

    factors *= counts_desired / counts_observed

    sinogram_support = sinogram.ufuncs.greater(0)
    smoothed_support = Y.element(
            [sc_gaussian(sino_support, sigma=[1, 2 / X.cell_sides[0]])
             for sino_support in sinogram_support])
    background = 10 * smoothed_support + 10
    background *= counts_background / background.ufuncs.sum()
    data = odl.phantom.poisson_noise(factors * sinogram + background,
                                     seed=1807)

    np.save(file_data, (data, factors, background))

    fig1 = plt.figure(1)
    groundtruth.show('groundtruth', clim=clim, fig=fig1)
    fig2 = plt.figure(2)
    fig2.clf()
    i = 0
    plt.plot((sinogram[i]).asarray()[0], label='G(x)')
    plt.plot((factors[i]*sinogram[i]).asarray()[0], label='factors * G(x)')
    plt.plot(data[i].asarray()[0], label='data')
    plt.plot(background[i].asarray()[0], label='background')
    plt.legend()

    fig1.savefig('{}/groundtruth.png'.format(folder_main), bbox_inches='tight')
    fig2.savefig('{}/components1D.png'.format(folder_main),
                 bbox_inches='tight')

else:
    (data, factors, background) = np.load(file_data)
    data = Y.element(data)
    factors = Y.element(factors)
    background = Y.element(background)

# data fit
f = odl.solvers.SeparableSum(
        *[spdhg.KullbackLeiblerSmooth(Yi, yi, ri)
          for Yi, yi, ri in zip(Y, data, background)])
# TODO: should be like:
# f = spdhg.KullbackLeiblerSmooth(Y, data).translated(-background)

# prior and regularisation parameter
g = spdhg.TV_NonNegative(X, alpha=5e-2, strong_convexity=5e-1)

# operator
A = odl.BroadcastOperator(*[fi * Gi for fi, Gi in zip(factors, G)])

# objective functional
obj_fun = f * A + g

# rho is the square root of the upper bound of the step size constraint
rho = 0.99

# define strong convexity constants
mu_i = [fi.convex_conj.strong_convexity for fi in f]
mu_f = np.min(mu_i)
mu_g = g.strong_convexity

#%%
# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not spdhg.exists(file_target):
    file_normA = '{}/norms_{}subsets.npy'.format(folder_main, 1)
    if not spdhg.exists(file_normA):
        # compute norm of operator
        normA = [tol_norm * A.norm(estimate=True)]

        np.save(file_normA, normA)

    else:
        normA = np.load(file_normA)

    # set step size parameters
    kappa = np.sqrt(1 + normA[0]**2 / (mu_g * mu_f) / rho**2)
    sigma = 1 / ((kappa - 1) * mu_f)
    tau = 1 / ((kappa - 1) * mu_g)
    theta = 1 - 2 / (1 + kappa)

    # initialise variables
    x_opt = X.zero()
    y_opt = Y.zero()

    # create callback
    callback = (odl.solvers.CallbackPrintIteration(step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(step=10, cumulative=True))

    # compute a saddle point with PDHG and time the reconstruction
    g.prox_options['p'] = None
    odl.solvers.pdhg(x_opt, f, g, A, tau, sigma, n_iter_target, y=y_opt,
                     theta=theta, callback=callback)

    # rescale image and save
    x = (x_opt - clim[0]) / (clim[1] - clim[0])
    x = np.minimum(np.maximum(x, -1), 1)
    sk_imsave('{}/x_opt.png'.format(folder_main), x)

    # compute the subgradients of the saddle point
    q_opt = -A.adjoint(y_opt)
    r_opt = A(x_opt)
    # compute the objective function value at the saddle point
    obj_opt = obj_fun(x_opt)

    # save saddle point
    np.save(file_target, (x_opt, y_opt, q_opt, r_opt, obj_opt))

    # show saddle point and subgradients
    fig1 = plt.figure(1)
    plt.clf()
    x_opt.show('x saddle', clim=clim, fig=fig1)

    # choose
    i = 0

    fig2 = plt.figure(2)
    plt.clf()
    y_opt[i].show('y saddle[{}]'.format(i), fig=fig2)

    fig3 = plt.figure(3)
    plt.clf()
    x = q_opt.ufuncs.absolute().ufuncs.max()
    q_opt.show('q saddle', fig=fig3, clim=[-x, x])

    fig4 = plt.figure(4)
    plt.clf()
    r_opt[i].show('r saddle[{}]'.format(i), fig=fig4)

    # save images
    fig1.savefig('{}/x_opt_fig.png'.format(folder_main), bbox_inches='tight')
    fig2.savefig('{}/y_opt.png'.format(folder_main), bbox_inches='tight')
    fig3.savefig('{}/q_opt.png'.format(folder_main), bbox_inches='tight')
    fig4.savefig('{}/r_opt.png'.format(folder_main), bbox_inches='tight')

else:
    (x_opt, y_opt, q_opt, r_opt, obj_opt) = np.load(file_target)

    x_opt = X.element(x_opt)
    y_opt = Y.element(y_opt)
    q_opt = X.element(q_opt)
    r_opt = Y.element(r_opt)

# set verbosity level
verbose = 1

# set norms of the primal and dual variable
norm_x_sq = 1 / 2 * odl.solvers.L2NormSquared(X)
norm_y_sq = 1 / 2 * odl.solvers.L2NormSquared(Y)

dist_x = norm_x_sq.translated(x_opt)
dist_y = norm_y_sq.translated(y_opt)


# define callback to store function values
class CallbackStore(odl.solvers.Callback):

    def __init__(self, iter_save_data):
        self.iter_save_data = iter_save_data
        self.iteration_count = 0
        self.out = []

    def __call__(self, x, **kwargs):

        if self.iteration_count in self.iter_save_data:
            obj = obj_fun(x[0])

            dx = dist_x(x[0])
            dy = dist_y(x[1])
            d = dx + dy

            self.out.append({'obj': obj,
                             'dist': d, 'dist_x': dx, 'dist_y': dy,
                             'iter': self.iteration_count})

        self.iteration_count += 1


# number of subsets for each algorithm
n_subsets = {}
n_subsets['pdhg'] = 1
n_subsets['spdhg_uni10'] = 10
n_subsets['spdhg_uni50'] = 50
n_subsets['spdhg_uni200'] = 200

# number of iterations for each algorithm
n_iter = {}
iter_save_data = {}
for alg in n_subsets.keys():
    n_iter[alg] = n_epochs * n_subsets[alg]
    iter_save_data[alg] = range(0, n_iter[alg] + 1, n_subsets[alg])

# %%
# run algorithms
for alg in ['pdhg', 'spdhg_uni10', 'spdhg_uni50', 'spdhg_uni200']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob = None
    extra = None
    sigma = None
    tau = None
    theta = None
    factr = None

    # create lists for subset division
    n = n_subsets[alg]
    (subset2ind, ind2subset) = spdhg.divide_1Darray_equally(range(len(A)), n)

    # set random seed so that results are reproducable
    np.random.seed(1807)

    file_normA = '{}/norms_{:.0f}subsets.npy'.format(folder_main, n)
    if not spdhg.exists(file_normA):
        A_subsets = [odl.BroadcastOperator(*[A[i] for i in subset])
                     for subset in subset2ind]
        normA = [tol_norm * Ai.norm(estimate=True) for Ai in A_subsets]
        np.save(file_normA, normA)

    else:
        normA = np.load(file_normA)

    # choose parameters for algorithm
    if alg == 'pdhg':
        kappa = np.sqrt(1 + normA[0]**2 / (mu_g * mu_f) / rho**2)
        prob_subset = [1]
        prob = [1] * Y.size
        sigma = [1 / ((kappa - 1) * mu_f)] * Y.size
        tau = 1 / ((kappa - 1) * mu_g)
        theta = 1 - 2 / (1 + kappa)

    elif alg in ['spdhg_uni10', 'spdhg_uni50', 'spdhg_uni200']:
        kappa = [np.sqrt(1 + normAi**2 / (mu_g * mui) / rho**2)
                 for normAi, mui in zip(normA, mu_i)]
        kappa_max = max(kappa)
        prob_subset = [1 / n] * n
        prob = [1 / n] * Y.size
        sigma = [1 / ((kappa_max - 1) * mui) for mui in mu_i]
        tau = 1 / ((n * kappa_max + n - 2) * mu_g)
        theta = 1 - 2 / (n + n * kappa_max)

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

    g.prox_options['p'] = None

    callback([x, y])

    spdhg.spdhg(x, f, g, A, tau, sigma, n_iter[alg], prob, fun_select, y=y,
                theta=theta, callback=callback)

    output = callback.callbacks[1].out

    np.save('{}/{}_{}_output'.format(folder_npy, filename, alg),
            (iter_save_data[alg], n_iter[alg], x, output, n_subsets[alg],
             theta))

# %%
algorithms = ['pdhg', 'spdhg_uni10', 'spdhg_uni50', 'spdhg_uni200']
iter_save_data_v = {}
n_iter_v = {}
image_v = {}
output_v = {}
n_subsets_v = {}
theta_v = {}
for alg in algorithms:
    (iter_save_data_v[alg], n_iter_v[alg], image_v[alg], output_v[alg],
     n_subsets_v[alg], theta_v[alg]) = np.load(
             '{}/{}_{}_output.npy'.format(folder_npy, filename, alg))

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

    x = (image_v[alg] - clim[0]) / (clim[1] - clim[0])
    x = np.minimum(np.maximum(x, -1), 1)
    sk_imsave('{}/{}_{}_image.png'.format(folder_today, filename, alg), x)

    image_v[alg].show(alg, clim=clim, fig=fig)
    fig.savefig('{}/{}.png'.format(folder_today, alg), bbox_inches='tight')

markers = plt.Line2D.filled_markers

all_plots = ['dist', 'dist_x', 'dist_y', 'obj', 'obj_rel']
logy_plot = all_plots

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
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 4)
color = bmap.mpl_colors
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
marker = ('o', 'v', 's', 'p', 'd', 'x', 'x')  # set markers
mevery = [(i / 30., .1) for i in range(20)]  # how many markers to draw
msize = 9  # marker size

fig = []

# ### draw first figure
fig.append(plt.figure(1))
plt.clf()
label = ['PDHG', 'SPDHG (10 subsets)', 'SPDHG (50)', 'SPDHG (200)']
meas = 'dist'
for k, alg in enumerate(['pdhg', 'spdhg_uni10', 'spdhg_uni50',
                         'spdhg_uni200']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas] / output_resorted[alg][meas][0]
    plt.semilogy(x, y, color=color[k], linestyle=lstyle, linewidth=lwidth,
                 marker=marker[k], markersize=msize, markevery=mevery[k],
                 label=label[k])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative distance to saddle point')
plt.gca().yaxis.set_ticks(np.logspace(-6, 0, 3))

# set x and y limits
plt.ylim((1e-6, 2))
plt.xlim((0, 30))

# create legend
plt.legend(numpoints=1, loc='best', ncol=1)

###
fig.append(plt.figure(2))
plt.clf()

max_iter = 100
y_min = 1e-7
meas = 'obj_rel'
label = ['PDHG', 'SPDHG (10 subsets)', 'SPDHG (50)', 'SPDHG (200)']
for k, alg in enumerate(['pdhg', 'spdhg_uni10', 'spdhg_uni50',
                         'spdhg_uni200']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    i = np.less_equal(x, max_iter+1) & np.greater_equal(y, y_min)
    plt.semilogy(x[i], y[i], color=color[k], linestyle=lstyle,
                 linewidth=lwidth, marker=marker[k], markersize=msize,
                 markevery=mevery[k], label=label[k])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative objective')
plt.gca().yaxis.set_ticks(np.logspace(-6, 0, 3))

# set x and y limits
plt.ylim((y_min, 2))
plt.xlim((0, 30))

# create legend
plt.legend(numpoints=1, loc='best', ncol=1)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/{}_output{}.png'.format(folder_today, filename, i),
               bbox_inches='tight')
