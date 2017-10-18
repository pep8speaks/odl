# ----------------------------------------------------------------------
# Matthias J. Ehrhardt
# Cambridge Image Analysis, University of Cambridge, UK
# m.j.ehrhardt@damtp.cam.ac.uk
#
# Copyright 2016-2017 University of Cambridge
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

"""Primal-Dual Hybrid Gradient (PDHG) algorithms"""

from __future__ import print_function
import numpy as np

__author__ = 'Matthias J. Ehrhardt'
__copyright__ = 'Copyright 2016-2017, University of Cambridge'
__all__ = ('pdhg', 'spdhg', 'pa_spdhg', 'spdhg_generic', 'da_spdhg',
           'spdhg_pesquet')


def pdhg(x, f, g, A, tau, sigma, niter, **kwargs):
    """Computes a saddle point with PDHG.

    This algorithm is the same as "algorithm 1" in [CP2011a] but with
    extrapolation on the dual variable.


    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : function
        Functional Y -> IR_infty that has a convex conjugate with a
        proximal operator, i.e. f.convex_conj.proximal(sigma) : Y -> Y.
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : function
        Operator A : X -> Y that posseses an adjoint: A.adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    theta : scalar
        Extrapolation factor.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CP2011a] Chambolle, A and Pock, T. *A First-Order
    Primal-Dual Algorithm for Convex Problems with Applications to
    Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011),
    pp 120-145.
    """

    # Dual variable
    y = kwargs.pop('y', None)

    def fun_select(k): return [0]

    y_list = [y]

    spdhg_generic(x, [f], g, [A], tau, [sigma], niter, fun_select, y=y_list,
                  **kwargs)

    y.assign(y_list[0])


def spdhg(x, f, g, A, tau, sigma, niter, prob, fun_select, **kwargs):
    """Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that posses adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations
    prob: list
        List of probabilities that an index i is selected each iteration.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}.

    Other Parameters
    ----------------
    y : dual variable, optional
        Dual variable is part of a product space. By default equals 0.
    z : variable, optional
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    theta : scalar
        Global extrapolation factor.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P., and
    Schoenlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] Ehrhardt, M. J., Markiewicz, P. J., Richtárik, P., Schott, J.,
    Chambolle, A. and Schoenlieb, C.-B. *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """

    # Dual variable
    y = kwargs.pop('y', None)

    extra = [1 / p for p in prob]

    spdhg_generic(x, f, g, A, tau, sigma, niter, fun_select, y=y, extra=extra,
                  **kwargs)


def pa_spdhg(x, f, g, A, tau, sigma, niter, prob, mu_g, fun_select, **kwargs):
    """Computes a saddle point with a stochastic PDHG and primal acceleration.

    Next to other standard arguments, this algorithm requires the strong
    convexity constant mu_g of g.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that posses adjoints: A[i].adjoint
    tau : scalar
        Step size for primal variable.
    sigma : scalar
        Step size for dual variable.
    niter : int
        Number of iterations
    prob: list
        List of probabilities that an index i is selected each iteration.
    mu_g : scalar
        Strong convexity constant of g.
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}.

    Other Parameters
    ----------------
    y : dual variable, optional
        Dual variable is part of a product space. By default equals 0.
    z : variable, optional
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P., and
    Schoenlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """

    # Dual variable
    y = kwargs.pop('y', None)

    extra = [1 / p for p in prob]

    spdhg_generic(x, f, g, A, tau, sigma, niter, fun_select, extra=extra,
                  mu_g=mu_g, y=y, **kwargs)


def spdhg_generic(x, f, g, A, tau, sigma, niter, fun_select, **kwargs):
    """Computes a saddle point with a stochastic PDHG.

    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that

    (x*, y*) in arg min_x max_y sum_i=1^n <y_i, A_i> - f*[i](y_i) + g(x)

    where g : X -> IR_infty and f[i] : Y[i] -> IR_infty are convex, l.s.c. and
    proper functionals. For this algorithm, they all may be non-smooth and no
    strong convexity is assumed.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that posses adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : scalar
        Scalar / vector / matrix used as step size for dual variable. Note that
        the proximal operator related to f (see above) has to be well-defined
        for this input.
    niter : int
        Number of iterations
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}.

    Other Parameters
    ----------------
    y : dual variable, optional
        Dual variable is part of a product space. By default equals 0.
    z : variable, optional
        Adjoint of dual variable, z = A^* y. By default equals 0 if y = 0.
    mu_g : scalar
        Strong convexity constant of g.
    theta : scalar
        Global extrapolation factor.
    extra : list
        List of local extrapolation paramters for every index i.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P., and
    Schoenlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957 (2017).

    [E+2017] Ehrhardt, M. J., Markiewicz, P. J., Richtárik, P., Schott, J.,
    Chambolle, A. and Schoenlieb, C.-B. *Faster PET reconstruction with a
    stochastic primal-dual hybrid gradient method*. Wavelets and Sparsity XVII,
    58 (2017) http://doi.org/10.1117/12.2272946.
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None and y.norm() == 0:
        z = A.domain.zero()

    # Strong convexity of g
    mu_g = kwargs.pop('mu_g', None)
    if mu_g is None:
        update_proximal_primal = False
    else:
        update_proximal_primal = True

    # Global extrapolation factor theta
    theta = kwargs.pop('theta', 1)

    # Extrapolation factor theta
    extra = kwargs.pop('extra', None)
    if extra is None:
        extra = [1 for _ in sigma]

    # Initialize variables
    z_relax = z.copy()
    dual_tmp = A.range.element()
    dz = z.copy()

    # Save proximal operators
    proximal_dual_sigma = [fi.convex_conj.proximal(si)
                           for fi, si in zip(f, sigma)]
    proximal_primal_tau = g.proximal(tau)

    # run the iterations
    for k in range(niter):

        # select block
        selected = fun_select(k)

        # update primal variable
        z_relax.lincomb(1, x, -tau, z_relax)  # z_relax used as tmp variable
        proximal_primal_tau(z_relax, out=x)

        # update extrapolation parameter theta
        if update_proximal_primal:
            theta = 1 / np.sqrt(1 + 2 * mu_g * tau)

        # update dual variable and adj_y, adj_y_bar
        z_relax.assign(z)
        for i in selected:

            # update dual variable
            dual_tmp[i].assign(y[i])
            A[i](x, out=y[i])
            y[i].lincomb(1, dual_tmp[i], sigma[i], y[i])

            yi = y[i].copy()
            proximal_dual_sigma[i](yi, out=y[i])

            # update adjoint of dual variable
            A[i].adjoint(y[i] - dual_tmp[i], out=dz)
            z += dz

            # compute extrapolation
            z_relax += (1 + extra[i] * theta) * dz

        # update the step sizes tau and sigma for acceleration
        if update_proximal_primal:
            for i in range(len(sigma)):
                sigma[i] /= theta
            tau *= theta

            proximal_dual_sigma = [fi.convex_conj.proximal(si)
                                   for fi, si in zip(f, sigma)]
            proximal_primal_tau = g.proximal(tau)

        if callback is not None:
            callback([x, y])


def da_spdhg(x, f, g, A, tau, sigma_tilde, niter, extra, prob, mu, fun_select,
             **kwargs):
    """Computes a saddle point with a PDHG and dual acceleration.

    It therefore requires the functionals f*_i to be mu[i] strongly convex.

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that posses adjoints: A[i].adjoint
    tau : scalar
        Initial step size for primal variable.
    sigma_tilde : scalar
        Related to initial step size for dual variable.
    niter : int
        Number of iterations
    extra: list
        List of local extrapolation paramters for every index i.
    prob: list
        List of probabilities that an index i is selected each iteration.
    mu: list
        List of strong convexity constants of f*, i.e. mu[i] is the strong
        convexity constant of f*[i].
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}.

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [CERS2017] Chambolle, A., Ehrhardt, M. J., Richtárik, P., and
    Schoenlieb, C.-B. *Stochastic Primal-Dual Hybrid Gradient Algorithm with
    Arbitrary Sampling and Imaging Applications*.
    ArXiv: http://arxiv.org/abs/1706.04957 (2017).
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None and y.norm() == 0:
        z = A.domain.zero()

    # Initialize variables
    z_relax = z.copy()
    dual_tmp = y.copy()
    dz = z.copy()

    # run the iterations
    for k in range(niter):
        # select block
        selected = fun_select(k)

        # update extrapolation parameter theta
        theta = 1 / np.sqrt(1 + 2 * sigma_tilde)
        # update primal variable
        g.proximal(tau)(x - tau * z, out=x)

        # update dual variable and adj_y, adj_y_bar
        z_relax.assign(z)
        for i in selected:
            # compute the step sizes sigma_i based on sigma_tilde
            sigma_i = sigma_tilde / (
                    mu[i] * (prob[i] - 2 * (1 - prob[i]) * sigma_tilde))

            # update dual variable
            dual_tmp[i].assign(y[i])
            A[i](x, out=y[i])
            y[i].lincomb(1, dual_tmp[i], sigma_i, y[i])
            yi = y[i].copy()
            f[i].convex_conj.proximal(sigma_i)(yi, out=y[i])

            # update adjoint of dual variable
            A[i].adjoint(y[i] - dual_tmp[i], out=dz)
            z += dz

            # compute extrapolation
            z_relax += (1 + theta * extra[i]) * dz

        # update the step sizes tau and sigma_tilde for acceleration
        sigma_tilde *= theta
        tau /= theta

        if callback is not None:
            callback([x, y])


def spdhg_pesquet(x, f, g, A, tau, sigma, niter, fun_select, **kwargs):
    """Computes a saddle point with a stochstic variant of PDHG [PR2015].

    Parameters
    ----------
    x : primal variable
        This variable is both input and output of the method.
    f : functions
        Functionals Y[i] -> IR_infty that all have a convex conjugate with a
        proximal operator, i.e.
        f[i].convex_conj.proximal(sigma[i]) : Y[i] -> Y[i].
    g : function
        Functional X -> IR_infty that has a proximal operator, i.e.
        g.proximal(tau) : X -> X.
    A : functions
        Operators A[i] : X -> Y[i] that posses adjoints: A[i].adjoint
    tau : scalar / vector / matrix
        Step size for primal variable. Note that the proximal operator of g
        has to be well-defined for this input.
    sigma : list
        List of scalars / vectors / matrices used as step sizes for the dual
        variable. Note that the proximal operators related to f (see above)
        have to be well-defined for this input.
    niter : int
        Number of iterations
    fun_select : function
        Function that selects blocks at every iteration IN -> {1,...,n}.

    Other Parameters
    ----------------
    y: dual variable
        Dual variable is part of a product space
    z: variable
        Adjoint of dual variable, z = A^* y.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    References
    ----------
    [PR2015] Pesquet, J.-C., & Repetti, A. *A Class of Randomized Primal-Dual
    Algorithms for Distributed Optimization*.
    ArXiv: http://arxiv.org/abs/1406.6404 (2015).
    """

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = A.range.zero()

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None and y.norm() == 0:
        z = A.domain.zero()

    # Save proximal operators
    prox_f = [f[i].convex_conj.proximal(sigma[i]) for i in range(len(sigma))]
    prox_g = g.proximal(tau)

    # Initialize variables
    x_relax = x.copy()
    primal_tmp = A.domain.element()
    dual_tmp = A.range.element()
    dz = z.copy()

    # run the iterations
    for k in range(niter):
        x_relax.lincomb(-1, x)

        # update primal variable
        primal_tmp.lincomb(1, x, -tau, z)
        prox_g(primal_tmp, out=x)

        # compute extrapolation
        x_relax.lincomb(1, x_relax, 2, x)

        # select block
        selected = fun_select(k)

        # update dual variable and adj_y
        for i in selected:
            # update dual variable
            dual_tmp[i].assign(y[i])

            yi = y[i] + sigma[i] * A[i](x_relax)
            prox_f[i](yi, out=y[i])

            # update adjoint of dual variable
            dz = A[i].adjoint(y[i] - dual_tmp[i])
            z += dz

        if callback is not None:
            callback([x, y])
