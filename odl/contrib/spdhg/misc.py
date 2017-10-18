# ----------------------------------------------------------------------
# Matthias J. Ehrhardt
# Cambridge Image Analysis, University of Cambridge, UK
# m.j.ehrhardt@damtp.cam.ac.uk
#
# Copyright 2016 University of Cambridge
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

"""Functions for folders and files."""

from __future__ import print_function
import os
import numpy as np
import odl
from scipy.signal import convolve2d as scipy_convolve2d

__author__ = "Matthias J. Ehrhardt"
__copyright__ = "Copyright 2016, University of Cambridge"
__all__ = ('exists', 'mkdir', 'TV', 'TV_NonNegative', 'bregman',
           'divide_1Darray_equally', 'Blur', 'HuberL1',
           'KullbackLeiblerSmooth')


def exists(path):
    return os.path.exists(path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def bregman(f, v, r):
    return (odl.solvers.FunctionalQuadraticPerturb(f, linear_term=-r)
            - f(v) + r.inner(v))


def divide_1Darray_equally(ind, n_subsets):
    """
    Divide an array into equal chunks to be used for instance in OSEM.

    Parameters
    ----------
    ind : ndarray
        input array
    nsubsets : int
        number of subsets to be divided into

    Returns
    -------
    subset2ind : list
        list of indices for each subset
    ind2subset : list
        list of subsets for each index
    """

    n_ind = len(ind)
    subset2ind = []

    for i in range(n_subsets):
        subset2ind.append([])
        for j in range(i, n_ind, n_subsets):
            subset2ind[i].append(ind[j])

    ind2subset = []
    for i in range(n_ind):
        ind2subset.append([])

    for i in range(n_subsets):
        for j in subset2ind[i]:
            ind2subset[j].append(i)

    return (subset2ind, ind2subset)


def TV(domain, grad=None):
    """ Total variation functional.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    grad : gradient operator, optional
        Gradient operator of the total variation functional. This may be any
        linear operator and thereby generalizing TV. default=forward
        differences with Neumann boundary conditions

    Examples
    --------
    Check that the total variation of a constant is zero

    >>> import odl.contrib.spdhg as spdhg, odl
    >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
    >>> TV = spdhg.TV(space)
    >>> x = space.one()
    >>> TV(x) < 1e-10
    """

    if grad is None:
        grad = odl.Gradient(domain, method='forward', pad_mode='symmetric')
        grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
    else:
        grad = grad

    f = odl.solvers.GroupL1Norm(grad.range, exponent=2)

    return f * grad


class TV_NonNegative(odl.solvers.Functional):
    """ Total variation function with nonnegativity constraint and strongly
    convex relaxation.

    In formulas, this functional may represent

        alpha * |grad x|_1 + char_fun(x) + beta/2 |x|^2_2

    with regularization parameter alpha and strong convexity beta. In addition,
    the nonnegativity constraint is achieved with the characteristic function

        char_fun(x) = 0 if x >= 0 and infty else.

    Parameters
    ----------
    domain : odlspace
        domain of TV functional
    alpha : scalar, optional
        Regularization parameter, positive
    prox_options : dict, optional
        name: string, optional
            name of the method to perform the prox operator, default=FGP
        warmstart: boolean, optional
            Do you want a warm start, i.e. start with the dual variable
            from the last call? default=True
        niter: int, optional
            number of iterations per call, default=5
        p: array, optional
            initial dual variable, default=zeros
    grad : gradient operator, optional
        Gradient operator to be used within the total variation functional.
        default=see TV
    """

    def __init__(self, domain, alpha=1, prox_options={}, grad=None,
                 strong_convexity=0):
        """
        """

        self.strong_convexity = strong_convexity

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        self.prox_options = prox_options

        self.alpha = alpha
        self.TV = TV(domain, grad=grad)
        self.grad = self.TV.right
        self.NN = odl.solvers.IndicatorBox(domain, 0, np.inf)
        self.L2 = 0.5 * odl.solvers.L2NormSquared(domain)

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        """ Characteristic function of the non-negative orthant

        Parameters
        ----------
        x : np.array
            vector / image

        Returns
        -------
        extended float (with infinity)
            Is the input in the non-negative orthant?

        Examples
        --------
        Check that the total variation of a constant is zero

        >>> import odl.contrib.spdhg as spdhg, odl
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> TVNN = spdhg.TV_NonNegative(space, alpha=2)
        >>> x = space.one()
        >>> TVNN(x) < 1e-10

        Check that negative functions are mapped to infty

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> TVNN = spdhg.TV_NonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> np.isinf(TVNN(x))
        """

        nn = self.NN(x)

        if nn is np.inf:
            return nn
        else:
            out = self.alpha * self.TV(x) + nn
            if self.strong_convexity > 0:
                out += self.strong_convexity * self.L2(x)
            return out

    def proximal(self, sigma):
        """ Prox operator of TV. It allows the proximal step length to be a vector
        of positive elements.

        Parameters
        ----------
        x : np.array
            vector / image

        Returns
        -------
        extended float (with infinity)
            Is the input in the non-negative orthant?

        Examples
        --------
        Check that the proximal operator is the identity for sigma=0

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> TVNN = spdhg.TV_NonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = TVNN.proximal(0)(x)
        >>> (y-x).norm() < 1e-10

        Check that negative functions are mapped to 0

        >>> import odl.contrib.spdhg as spdhg, odl, numpy as np
        >>> space = odl.uniform_discr([0, 0], [3, 3], [3, 3])
        >>> TVNN = spdhg.TV_NonNegative(space, alpha=2)
        >>> x = -space.one()
        >>> y = TVNN.proximal(0.1)(x)
        >>> y.norm() < 1e-10
        """

        if sigma == 0:
            return odl.IdentityOperator(self.domain)

        else:
            def TV_prox(z, out=None):

                if out is None:
                    out = z.space.zero()

                opts = self.prox_options

                sigma_ = np.copy(sigma)
                z_ = z.copy()

                if self.strong_convexity > 0:
                    sigma_ /= (1 + sigma * self.strong_convexity)
                    z_ /= (1 + sigma * self.strong_convexity)

                def proj_C(x, out=None):
                    return self.NN.proximal(1)(x, out)

                def proj_P(x, out=None):
                    return odl.solvers.GroupL1Norm(
                            self.grad.range, exponent=2
                            ).convex_conj.proximal(0)(x, out=out)

                if opts['name'] == 'FGP':
                    if opts['warmstart']:
                        if opts['p'] is None:
                            opts['p'] = self.grad.range.zero()

                        p = opts['p']
                    else:
                        p = self.grad.range.zero()

                    sigmaSqrt = np.sqrt(sigma_)

                    z_ /= sigmaSqrt
                    grad = sigmaSqrt * self.grad
                    grad.norm = sigmaSqrt * self.grad.norm
                    niter = opts['niter']
                    alpha = self.alpha
                    out[:] = FGP_dual(p, z_, alpha, niter, grad, proj_C,
                                      proj_P, tol=opts['tol'])

                    out *= sigmaSqrt

                    return out

                else:
                    raise NotImplementedError('Not yet implemented')

            return TV_prox


def FGP_dual(p, data, alpha, n_iter, grad, proj_C, proj_P, tol=None, **kwargs):
    """ Computes a saddle point with the PDHG method.

    Parameters
    ----------
    p : np.array
        dual initial variable
    data : np.array
        noisy data / proximal point
    alpha : float
        regularization parameter
    n_iter : int
        number of iterations
    grad : gradient class
        class that supports grad(x), grad.adjoint(x), grad.norm
    proj_C : function
        projection onto the constraint set of the primal variable,
        e.g. non-negativity
    proj_P : function
        projection onto the constraint set of the dual variable,
        e.g. norm <= 1
    tol : float (optional)
        nonnegative parameter that gives the tolerance for convergence. If set
        None, then the algorithm will run for a fixed number of iterations

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    factr = 1 / (grad.norm**2 * alpha)

    q = p.copy()
    x = data.space.zero()

    t = 1.

    if tol is None:
        def convergenceEval(p1, p2):
            return False
    else:
        def convergenceEval(p1, p2):
            return (p1 - p2).norm() / p1.norm() < tol

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(n_iter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - alpha * x, out=x)
        grad(x, out=pnew)
        pnew *= factr
        pnew += q
        proj_P(pnew, out=pnew)

        converged = convergenceEval(p, pnew)

        if not converged:
            # update step size
            t = (1 + np.sqrt(1 + 4*t0**2))/2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1)/t * (pnew - p)

        p[:] = pnew

        if converged:
            t = None
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - alpha * grad.adjoint(p))

    return x


class Blur(odl.Operator):
    """Blur operator
    """

    def __init__(self, domain, kernel, boundary_condition='wrap'):
        """Initialize a new instance.
        """

        super().__init__(domain=domain, range=domain, linear=True)

        self.__kernel = kernel
        self.__boundary_condition = boundary_condition

    @property
    def kernel(self):
        return self.__kernel

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    def _call(self, x, out):
        out[:] = scipy_convolve2d(x, self.kernel, mode='same',
                                  boundary='wrap')

    @property
    def gradient(self):
        raise NotImplementedError('No yet implemented')

    @property
    def adjoint(self):
        adjoint_kernel = self.kernel.copy().conj()
        for i in range(len(adjoint_kernel.shape)):
            adjoint_kernel = np.flip(adjoint_kernel, i)

        return Blur(self.domain, adjoint_kernel, self.boundary_condition)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel,
            self.boundary_condition)


class HuberL1(odl.solvers.Functional):
    """The functional corresponding to the Huberized L1-norm.
    """

    def __init__(self, space, gamma):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        gamma : float
            Smoothing parameter of Huberization. If gamma = 0, then functional
            is non-smooth corresponds to the usual L1 norm. For gamma > 0, it
            has a 1/gamma-Lipschitz gradient so that its convex conjugate is
            gamma-strongly convex.

        Examples
        --------

        Compare HuberL1 and L1 for vanishing smoothing gamma=0

        >>> import numpy as np, odl odl.contrib.spdhg as spdhg
        >>> X = odl.uniform_discr([0, 0], [1, 1], [5, 5])
        >>> x = odl.phantom.white_noise(X)
        >>> alpha = float(np.random.rand(1))
        >>> gamma = 0
        >>> H = alpha * spdhg.HuberL1(X, gamma)
        >>> L1 = alpha * odl.solvers.L1Norm(X)
        >>> abs(H(x) - L1(x)) < 1e-10
        """
        self.gamma = float(gamma)
        self.strong_convexity = 0

        if self.gamma > 0:
            grad_lipschitz = 1 / self.gamma
        else:
            grad_lipschitz = np.inf

        super().__init__(space=space, linear=False,
                         grad_lipschitz=grad_lipschitz)

    def _call(self, x):
        '''Return the HuberL1-norm of ``x``.'''
        n = x.ufuncs.absolute()

        if self.gamma > 0:
            i = n.ufuncs.less(self.gamma)

#            n[i] = 1 / (2 * self.gamma) * n[i]**2 + self.gamma / 2

            n = i * (1 / (2 * self.gamma) * n**2 + self.gamma / 2)
            + i.ufuncs.logical_not() * n

        return self.domain.element(n).inner(self.domain.one())

    @property
    def convex_conj(self):
        '''The convex conjugate'''
        return HuberL1ConvexConj(self.domain, self.gamma)

    @property
    def proximal(self):
        '''The proximal operator'''
        raise NotImplementedError('Not yet implement. To be done.')

    @property
    def gradient(self):
        '''Gradient operator of the functional.'''
        raise NotImplementedError('Not yet implement. To be done.')

    def __repr__(self):
        '''Return ``repr(self)``.'''
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.domain,
                                       self.gamma)


class HuberL1ConvexConj(odl.solvers.Functional):
    """The convex conjugate of the huberized L1-norm.
    """

    def __init__(self, space, gamma):
        '''Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        gamma : float
            Smoothing parameter of Huberization. If gamma = 0, then functional
            is non-smooth corresponds to the usual L1 norm. For gamma > 0, it
            has a 1/gamma-Lipschitz gradient so that its convex conjugate is
            gamma-strongly convex.
        '''
        self.gamma = float(gamma)
        self.strong_convexity = self.gamma
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def _call(self, x):
        '''Return the Lp-norm of ``x``.'''
        f = (odl.solvers.L1Norm(self.domain).convex_conj +
             self.gamma / 2 * odl.solvers.L2NormSquared(self.domain))

        return f(x)

    @property
    def convex_conj(self):
        '''The convex conjugate'''
        return HuberL1(self.domain, self.gamma)

    @property
    def proximal(self):
        '''The proximal operator'''

        def ProxHuberL1_convexconj(sigma):
            return (odl.solvers.L1Norm(self.domain).convex_conj.proximal(0) *
                    1 / (1 + self.gamma * sigma))

        return ProxHuberL1_convexconj

    @property
    def gradient(self):
        '''Gradient operator of the functional.'''
        raise NotImplementedError('Not yet implemented.')

    def __repr__(self):
        '''Return ``repr(self)``.'''
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.domain,
                                       self.gamma)


class KullbackLeiblerSmooth(odl.solvers.Functional):

    """The smooth Kullback-Leibler divergence functional.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    smooth Kullback-Leibler functional :math:`\\phi` is defined as

    .. math::
        \\phi(x) = \\sum_{i=1}^n \\begin{cases}
                x + r - y + y * \\log(y / (x + r))
                    & \\text{if $x \geq 0$} \\
                (y / (2 * r^2)) * x^2 + (1 - y / r) * x + r - b +
                    b * \\log(b / r) & \\text{else}
                                 \\end{cases}

    where all variables on the right hand side of the equation have a subscript
    i which is omitted for readability.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P. and
    Schönlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957, 2017
    """

    def __init__(self, space, data, background):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        data : ``space`` `element-like`
            Data vector which has to be non-negative.
        background : ``space`` `element-like`
            Background vector which has to be non-negative.
        """

        self.strong_convexity = 0

        if background.ufuncs.less_equal(0).ufuncs.sum() > 0:
            raise NotImplementedError('Background must be positive')

        super().__init__(space=space, linear=False,
                         grad_lipschitz=np.max(data/background**2))

        if data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        self.__data = data
        self.__background = background

    @property
    def data(self):
        """The data in the Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The background in the Kullback-Leibler functional."""
        return self.__background

    def _call(self, x):
        """Return the KL-diveregnce in the point ``x``.

        If any components of ``x`` is non-positive, the value is positive
        infinity.
        """
        y = self.data
        r = self.background

        # TODO: implement more efficiently in terms of memory and CPU/GPU
        # TODO: cover properly the case y = 0

        # x + r - y + y * log(y / (x + r)) = x - y * log(x + r) + c1
        # with c1 = r - y + y * log y
        i = x.ufuncs.greater_equal(0)
        obj = (i * (x + r - y + y * (y / (x + r)).ufuncs.log()))

        # (y / (2 * r^2)) * x^2 + (1 - y / r) * x + r - b + b * log(b / r)
        # = (y / (2 * r^2)) * x^2 + (1 - y / r) * x + c2
        # with c2 = r - b + b * log(b / r)
        i = i.ufuncs.logical_not()
        obj += i * (y / (2 * r**2) * x**2 + (1 - y / r) * x + r - y +
                    y * (y / r).ufuncs.log())

        return obj.inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of the functional.
        """
        raise NotImplementedError('No yet implemented')

    @property
    def proximal(self):
        """Return the `proximal factory` of the functional.
        """
        raise NotImplementedError('No yet implemented')

    @property
    def convex_conj(self):
        """The convex conjugate functional of the KL-functional."""
        return KullbackLeiblerSmoothConvexConj(self.domain, self.data,
                                               self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.data, self.background)


class KullbackLeiblerSmoothConvexConj(odl.solvers.Functional):

    """The convex conjugate of the smooth Kullback-Leibler divergence functional.

    Notes
    -----
    If the functional is defined on an :math:`\mathbb{R}^n`-like space, the
    convex conjugate of the smooth Kullback-Leibler functional :math:`\\phi^*`
    is defined as

    .. math::
        \\phi^*(x) = \\sum_{i=1}^n \\begin{cases}
                r^2 / (2 * y) * x^2 + (r - r^2 / y) * x + r^2 / (2 * y) +
                    3 / 2 * y - 2 * r - y * log(y / r)
                    & \\text{if $x < 1 - y / r$} \\
                - r * x - y * log(1 - x)
                    & \\text{if $1 - y / r <= x < 1} \\
                - + \infty
                    & \\text{else}
                                 \\end{cases}

    where all variables on the right hand side of the equation have a subscript
    i which is omitted for readability.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P. and
    Schönlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957, 2017
    """

    def __init__(self, space, data, background):
        '''Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` or `FnBase`
            Domain of the functional.
        data : ``space`` `element-like`
            Data vector which has to be non-negative.
        background : ``space`` `element-like`
            Background vector which has to be non-negative.
        '''

        # TODO: cover properly the case data = 0

        if background.ufuncs.less_equal(0).ufuncs.sum() > 0:
            raise NotImplementedError('Background must be positive')

        super().__init__(space=space, linear=False,
                         grad_lipschitz=np.inf)

        if data is not None and data not in self.domain:
            raise ValueError('`data` not in `domain`'
                             ''.format(data, self.domain))

        self.__data = data
        self.__background = background

        self.strong_convexity = np.min(self.background**2 / self.data)

    @property
    def data(self):
        """The data in the Kullback-Leibler functional."""
        return self.__data

    @property
    def background(self):
        """The background in the Kullback-Leibler functional."""
        return self.__background

    def _call(self, x):
        """Return the value in the point ``x``.

        If any components of ``x`` is larger than or equal to 1, the value is
        positive infinity.
        """

        # TODO: implement more efficiently in terms of memory and CPU/GPU
        # TODO: cover properly the case data = 0

        y = self.data
        r = self.background

        # if any element is greater or equal to one
        if x.ufuncs.greater_equal(1).ufuncs.sum() > 0:
            return np.inf

        # if x < 1 - y / r, then
        # r^2 / (2 * y) * x^2 + (r - r^2 / y) * x + r^2 / (2 * y) +
        # 3 / 2 * y - 2 * r - y * log(y / r)
        i = x.ufuncs.less(1 - y / r)
        obj = i * (r**2 / (2 * y) * x**2 + (r - r**2 / y) * x +
                   r**2 / (2 * y) + 3 / 2 * y - 2 * r -
                   y * (y / r).ufuncs.log())

        # if 1 - y / r <= x, then
        # r * x - y * log(1 - x)
        i = i.ufuncs.logical_not()
        obj += i * (-r * x - r * (1 - x).ufuncs.log())

        return obj.inner(self.domain.one())

    @property
    def gradient(self):
        """Gradient operator of the functional.
        """
        raise NotImplementedError('No yet implemented')

    @property
    def proximal(self):

        space = self.domain
        y = self.data
        r = self.background

        class ProxKullbackLeiblerSmooth_convexconj(odl.Operator):

            """Proximal operator of the convex conjugate of the smooth
            Kullback-Leibler functional.
            """

            def __init__(self, sigma):
                """Initialize a new instance.

                Parameters
                ----------
                sigma : positive float
                    Step size parameter
                """
                self.sigma = float(sigma)
                self.background = r
                self.data = y
                super().__init__(domain=space, range=space, linear=False)

            def _call(self, x, out=None):

                s = self.sigma
                y = self.data
                r = self.background

                # TODO: Make memory and CPU/GPU efficient
                # TODO: Make sure it works for data = 0

                if out is None:
                    out = self.domain.element()

                # if x < 1 - y / r, then
                # TODO: formulas missing
                i = np.less(x, 1 - self.data/self.background)
                out.assign(i * ((y * x - s * r * y + s * r**2) /
                                (y + s * r**2)))

                # else:
                # TODO: formulas missing
                i = i.ufuncs.logical_not()
                out += i * (0.5 * (x + s * r + 1 - ((x + s * r - 1)**2 +
                                                    4 * s * y).ufuncs.sqrt()))

                return out

        return ProxKullbackLeiblerSmooth_convexconj

    @property
    def convex_conj(self):
        """The convex conjugate functional of the smooth KL-functional."""
        return KullbackLeiblerSmooth(self.domain, self.data,
                                     self.background)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.data, self.background)
