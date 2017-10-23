"""Total variation denoising using PDHG.

Solve the L1-HuberTV problem

        min_{x >= 0} ||x - d||_1 + lam TV_gamma(x)

where ``grad`` the spatial gradient and ``d`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy
import odl
import matplotlib.pyplot as plt

# --- define setting --- #

# Read test image: use only every second pixel, convert integer to float
image = np.rot90(scipy.misc.ascent()[::2, ::2].astype('float'), 3)
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape)

# Create space element of ground truth
orig = space.element(image.copy())

# Add noise and convert to space element
noisy = odl.phantom.salt_pepper_noise(orig)

# Gradient operator
gradient = odl.Gradient(space, method='forward')

# regularization parameter
reg_param = 1

# l1 data matching
l1_norm = 1 / reg_param * odl.solvers.L1Norm(space).translated(noisy)

# HuberTV-regularization
huber_l1_norm = odl.solvers.HuberL1L2(gradient.range, gamma=.1)

# define objective
obj_fun = l1_norm + huber_l1_norm * gradient

# strong convexity of "f*"
strong_convexity = 1 / huber_l1_norm.grad_lipschitz


# define callback to store function values
class CallbackStore(odl.solvers.util.callback.Callback):
    def __init__(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.obj_function_values = []

    def __call__(self, x):
        self.iteration_count += 1
        self.iteration_counts.append(self.iteration_count)
        self.obj_function_values.append(obj_fun(x))

    def reset(self):
        self.iteration_count = 0
        self.iteration_counts = []
        self.obj_function_values = []


callback = (odl.solvers.CallbackPrintIteration() & CallbackStore())

# number of iterations
niter = 500

# Assign operator and functionals
op = gradient
f = huber_l1_norm
g = l1_norm

# The operator norm of the gradient with forward differences is well-known
op_norm = np.sqrt(8) + 1e-4

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Run algorithms 2 and 3
x = space.zero()
callback(x)
odl.solvers.pdhg(x, f, g, op, tau, sigma, niter, gamma_dual=strong_convexity,
                 callback=callback)
obj = callback.callbacks[1].obj_function_values

# %% Display results
# show images
clim = [0, 1]
cmap = 'gray'

orig.show('original', clim=clim, cmap=cmap)
noisy.show('noisy', clim=clim, cmap=cmap)
x.show('denoised', clim=clim, cmap=cmap)


# show convergence rate
def rel_fun(x):
    x = np.array(x)
    return (x - min(x)) / (x[0] - min(x))


i = np.array(callback.callbacks[1].iteration_counts)

plt.figure(1)
plt.clf()
plt.loglog(i, rel_fun(obj), label='alg')
plt.loglog(i[1:], 1. / i[1:], '--', label='$O(1/k)$')
plt.loglog(i[1:], 4. / i[1:]**2, ':', label='$O(1/k^2)$')
rho = 0.97
plt.loglog(i[1:], rho**i[1:], '-',
           label='$O(\\rho^k), \\rho={:3.2f}$'.format(rho))
plt.title('Function values')
plt.legend()
