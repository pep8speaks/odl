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

"""An example of using the SPDHG algorithm to solve a TV denoising problem
with Gaussian noise. We exploit the strong convexity of the data term to get
1/k^2 convergence on the primal part.

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
from scipy import misc as sc_misc
import odl
import brewer2mpl

__author__ = 'Matthias J. Ehrhardt'
__copyright__ = 'Copyright 2017, University of Cambridge'

# create folder for data TO BE CHANGED!!
folder_out = '.'
filename = 'example_spdhg_ROF_1k2_primal'

n_epochs = 300  # number of epochs
#n_epochs = 10  # number of epochs
n_iter_target = 2000

subfolder = '{}_{}epochs'.format('20171002_1400', n_epochs)

folder_main = '{}/{}'.format(folder_out, filename)
spdhg.mkdir(folder_main)

folder_today = '{}/{}'.format(folder_main, subfolder)
spdhg.mkdir(folder_today)

folder_npy = '{}/npy'.format(folder_today)
spdhg.mkdir(folder_npy)

# create ground truth
image_raw = np.rot90(sc_misc.imread('cms.jpg').astype('float32'), k=3)
image_gray = np.sum(image_raw, 2)

X = odl.uniform_discr([0, 0], image_gray.shape, image_gray.shape)
groundtruth = X.element(image_gray / np.max(image_gray))
clim = [0, 1]
cmap = 'gray'

# create data
data = odl.phantom.white_noise(X, mean=groundtruth, stddev=0.1, seed=1807)

# save images and data
if not spdhg.exists('{}/groundtruth.png'.format(folder_main)):
    fig1 = plt.figure(1)
    groundtruth.show(fig=fig1, clim=clim, cmap=cmap, title='ground truth')
    fig2 = plt.figure(2)
    data.show(fig=fig2, clim=clim, cmap=cmap, title='data')

    fig1.savefig('{}/groundtruth.png'.format(folder_main),
                 bbox_inches='tight')
    fig2.savefig('{}/data.png'.format(folder_main),
                 bbox_inches='tight')

# set regularisation parameter
alpha = .12

# gamma is the square root of the upper bound of the step size constraint
gamma = 1

# create forward operators
Dx = odl.PartialDerivative(X, 0, pad_mode='symmetric')
Dy = odl.PartialDerivative(X, 1, pad_mode='symmetric')
A = odl.BroadcastOperator(Dx, Dy)
Y = A.range

# set up functional f
f = odl.solvers.SeparableSum(*[odl.solvers.L1Norm(Yi) for Yi in Y])

# set up functional g
g = 1 / (2 * alpha) * odl.solvers.L2NormSquared(X).translated(data)

# define objective function
objFun = f * A + g

# define strong convexity constants
mu_g = 1 / alpha

# create target / compute a saddle point
file_target = '{}/target.npy'.format(folder_main)
if not spdhg.exists(file_target):

    # compute norm of operator
    normA = np.sqrt(8)

    # set step size parameters
    sigma = gamma / normA
    tau = gamma / normA

    # initialise variables
    x_opt = X.zero()
    y_opt = Y.zero()

    # compute a saddle point with PDHG and time the reconstruction
    callback = (odl.solvers.CallbackPrintIteration(step=10, end=', ') &
                odl.solvers.CallbackPrintTiming(step=10, cumulative=True))

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

# define Bregman distance for f and f_p
bregman_f = odl.solvers.SeparableSum(*[spdhg.bregman(fi.convex_conj, yi, ri)
                                       for fi, yi, ri in zip(f, y_opt, r_opt)])

# create distances for primal and dual variable
dist_x = norm_x_sq.translated(x_opt)
dist_y = norm_y_sq.translated(y_opt)


# define callback to store function values
class CallbackStore(odl.solvers.util.callback.Callback):

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

            self.out.append({'obj': obj, 'breg': breg, 'breg_x': breg_x,
                             'breg_y': breg_y, 'breg_erg': breg_erg,
                             'breg_erg_x': breg_erg_x,
                             'breg_erg_y': breg_erg_y,
                             'dist_x': dist_x(x[0]), 'dist_y': dist_y(x[1]),
                             'iter': k})

        self.iteration_count += 1


# number of subsets for each algorithm
n_subsets = dict()
n_subsets['pdhg'] = 1
n_subsets['pa_pdhg'] = 1
n_subsets['pesquet'] = 2
n_subsets['spdhg_uni2'] = 2
n_subsets['pa_spdhg_uni2'] = 2
n_subsets['odl'] = 1
n_subsets['pa_odl'] = 1

# number of iterations for each algorithm
n_iter = {}
iter_save_data = {}
for alg in n_subsets.keys():
    n_iter[alg] = n_epochs * n_subsets[alg]
    iter_save_data[alg] = range(0, n_iter[alg] + 1, n_subsets[alg])

#%%
# run algorithms
# TODO: ODL version to be included once the callback includes dual iterates
#for alg in ['pdhg', 'pesquet', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2', 'odl', 'pa_odl']:
for alg in ['pdhg', 'pesquet', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2']:
    print('======= ' + alg + ' =======')

    # clear variables in order not to use previous instances
    prob = None
    extra = None
    sigma = None
    tau = None
    theta = None

    # create lists for subset division
    n = n_subsets[alg]
    (subset2ind, ind2subset) = spdhg.divide_1Darray_equally(range(2), n)

    # set random seed so that results are reproducable
    np.random.seed(1807)

    # choose parameters for algorithm
    if alg == 'pdhg' or alg == 'pa_pdhg':
        prob_subset = [1] * n
        prob = [1] * Y.size
        sigma = [gamma / normA] * Y.size
        tau = gamma / normA

    elif alg == 'odl' or alg == 'pa_odl':
        sigma = gamma / normA
        tau = gamma / normA

    elif alg == 'pesquet':
        prob_subset = [1 / n] * n
        prob = [1 / n] * Y.size
        sigma = [gamma / normA] * Y.size
        tau = gamma / normA

    elif alg in ['spdhg_uni2'] or alg in ['pa_spdhg_uni2']:
        normA_i = [2] * n
        prob_subset = [1 / n] * n
        prob = [1 / n] * Y.size
        sigma = [gamma / nA for nA in normA_i]
        tau = gamma / (n * max(normA_i))

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
        spdhg.spdhg(x, f, g, A, tau, sigma, n_iter[alg], prob, y=y,
                    fun_select=fun_select, callback=callback)

    elif alg[:7] == 'pa_pdhg' or alg[:8] == 'pa_spdhg':
        spdhg.pa_spdhg(x, f, g, A, tau, sigma, n_iter[alg], prob, mu_g, y=y,
                       fun_select=fun_select, callback=callback)

    elif alg[:3] == 'odl':
        odl.solvers.pdhg(x, f, g, A, tau, sigma, n_iter[alg], y=y,
                         callback=callback)

    elif alg[:6] == 'pa_odl':
        odl.solvers.pdhg(x, f, g, A, tau, sigma, n_iter[alg], y=y,
                         callback=callback, gamma_primal=mu_g)

    elif alg == 'pesquet':
        spdhg.spdhg_pesquet(x, f, g, A, tau, sigma, n_iter[alg], y=y,
                            fun_select=fun_select, callback=callback)

    else:
        raise NameError('Algorithm not defined')

    output = callback.callbacks[1].out

    np.save('{}/{}_{}_output'.format(folder_npy, filename, alg),
            (iter_save_data[alg], n_iter[alg], x, output, n_subsets[alg]))

# %%
algorithms = ['pdhg', 'pa_pdhg', 'spdhg_uni2', 'pa_spdhg_uni2', 'pesquet']
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
             'breg_erg_y', 'breg', 'breg_y', 'breg_x']
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
bmap = brewer2mpl.get_map('Paired', 'Qualitative', 5)
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
meas = 'dist_x'
label = ['PDHG', 'SPDHG', 'PA-PDHG', 'PA-SPDHG']
for j, alg in enumerate(['pdhg', 'spdhg_uni2', 'pa_pdhg', 'pa_spdhg_uni2']):
    x = epochs_save_data[alg][1:]
    y = output_resorted[alg][meas][1:]
    plt.loglog(x, y, color=colors[j], linestyle=lstyle, linewidth=lwidth,
               marker=marker[j], markersize=msize, markevery=mevery[j],
               label=label[j])

y = 5e+4 / np.array(iter_save_data_v[alg][1:])**2
plt.loglog(x, y, color='gray', linestyle=lstyle_help, linewidth=lwidth_help,
           label=r'$\mathcal O(1/K^2)$')

y = 5e+3 / np.array(iter_save_data_v[alg][1:])
plt.loglog(x, y, color='gray', linestyle='-', linewidth=lwidth_help,
           label=r'$\mathcal O(1/K)$')

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('primal distance')
plt.gca().yaxis.set_ticks(np.logspace(-1, 3, 3))

# set x and y limits
plt.ylim((3e-1, 3e+3))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# ### next figure
fig.append(plt.figure(2))
plt.clf()

max_iter = 300
min_y = 5e+0
meas = 'breg'
label = ['PDHG', 'SPDHG', 'PA-PDHG', 'PA-SPDHG']
for j, alg in enumerate(['pdhg', 'spdhg_uni2', 'pa_pdhg', 'pa_spdhg_uni2']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    i = np.less(x, max_iter+1) & np.greater(x, 0) & np.greater(y, min_y)
    plt.loglog(x[i], y[i], color=colors[j], linestyle=lstyle, linewidth=lwidth,
               marker=marker[j], markersize=msize, markevery=mevery[j],
               label=label[j])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('Bregman distance')
plt.gca().yaxis.set_ticks(np.logspace(0, 4, 3))

# set x and y limits
plt.ylim((min_y, 1e+4))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# ### next figure
fig.append(plt.figure(3))
plt.clf()

max_iter = 400
min_y = 0
meas = 'obj_rel'
label = ['PDHG', 'SPDHG', 'PA-PDHG', 'PA-SPDHG']
for j, alg in enumerate(['pdhg', 'spdhg_uni2', 'pa_pdhg', 'pa_spdhg_uni2']):
    x = epochs_save_data[alg]
    y = output_resorted[alg][meas]
    i = np.less(x, max_iter+1) & np.greater(x, 0) & np.greater(y, min_y)
    plt.loglog(x[i], y[i], color=colors[j], linestyle=lstyle, linewidth=lwidth,
               marker=marker[j], markersize=msize, markevery=mevery[j],
               label=label[j])

# label x and y axis
plt.gca().set_xlabel('iterations [epochs]')
plt.gca().set_ylabel('relative objective')
plt.gca().yaxis.set_ticks(np.logspace(-5, -1, 3))

# set x and y limits
plt.ylim((1e-5, 2e-1))
plt.xlim((1, 300))

# create legend
legend = plt.legend(numpoints=1, loc='best', ncol=1, framealpha=0.0)

# %%
for i, fi in enumerate(fig):
    fi.savefig('{}/{}_output{}.png'.format(folder_today, filename, i),
               bbox_inches='tight')
