# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Operators based on (integral) transformations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy.signal as signal

from odl.discr.discr_ops import Resampling
from odl.discr.lp_discr import (
    DiscreteLp, DiscreteLpVector, uniform_discr, uniform_discr_fromdiscr)
from odl.operator.operator import Operator
from odl.set.sets import ComplexNumbers
from odl.trafos.fourier import FourierTransform
from odl.util.normalize import normalized_scalar_param_list, safe_int_conv


__all__ = ('Convolution', 'FourierSpaceConvolution', 'RealSpaceConvolution')


_REAL_CONV_SUPPORTED_IMPL = ('scipy_convolve',)
_FOURIER_CONV_SUPPORTED_IMPL = ('default', 'pyfftw')
_FOURIER_CONV_SUPPORTED_KER_MODES = ('real', 'fourier')


class Convolution(Operator):

    """Discretization of the convolution integral as an operator.

    This class provides common attributes and methods for subclasses
    implementing different variants.
    """

    def kernel(self):
        """Return the kernel of this convolution operator."""
        raise NotImplementedError('abstract method')


class FourierSpaceConvolution(Convolution):

    """Convolution implemented in Fourier space.

    This operator implements a discrete approximation to the continuous
    convolution with a fixed kernel.
    """

    def __init__(self, domain, kernel, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the operator.
        kernel :
            Convolution kernel of this operator. The kernel can be
            specified in several ways:

            domain `element-like` : The object is interpreted as the
            real-space kernel representation (mode ``'real'``).
            Valid for ``impl``: ``'numpy_ft', 'pyfftw_ft',
            'scipy_convolve'``

            `element-like` for the range of `FourierTransform` defined
            on ``dom`` : The object is interpreted as the Fourier
            transform of a real-space kernel (mode ``ft`` or ``'ft_hc'``).
            The correct space can be calculated with `reciprocal_space`.
            Valid for ``impl``: ``'default_ft', 'pyfftw_ft'``

            `array-like`, arbitrary length : The object is interpreted as
            real-space kernel (mode ``'real'``) and can be shorter than
            the convolved function.
            Valid for ``impl``: ``'scipy_convolve'``

        kernel_mode : {'real', 'fourier'}, optional
            Specifies the way the kernel is to be interpreted. If not
            provided, the kernel is tried to be converted into an element
            of a reasonable space derived from ``domain``.

            'real' : real-space kernel (default)

            'fourier' : fourier-space kernel

        impl : `str`, optional
            Implementation of the convolution. Available options are:

            'default' : Fourier transform using the default FFT
            implementation

            'pyfftw' : Fourier transform using pyFFTW (faster than
            the default FT)

        axes : sequence of `int`, optional
            Dimensions in which to convolve. Default: all axes

        scale : `bool`, optional
            If `True`, scale the discrete convolution by
            ``2*pi ** (ndim/2)``, such that it corresponds to a
            continuous convolution.
            Default: `True`

        kwargs :
            Extra arguments are passed to the transform when the
            kernel FT is calculated.

        See also
        --------
        FourierTransform : discretization of the continuous FT
        scipy.signal.convolve : real-space convolution

        Notes
        -----
        - The continuous convolution of two functions
          :math:`f` and :math:`g` defined on :math:`\mathbb{R^d}` is

              :math:`[f \\ast g](x) =
              \int_{\mathbb{R^d}} f(x - y) g(y) \mathrm{d}y`.

          With the help of the Fourier transform, this operation can be
          expressed as a multiplication,

              :math:`\\big[\mathcal{F}(f \\ast g)\\big](\\xi)
              = (2\pi)^{\\frac{d}{2}}\,
              \\big[\mathcal{F}(f)\\big](\\xi) \cdot
              \\big[\mathcal{F}(g)\\big](\\xi)`.

          This implementation covers the case of a fixed convolution
          kernel :math:`k`, i.e. the linear operator

              :math:`\\big[\mathcal{C}_k(f)\\big](x) = [k \\ast f](x)`.
        """
        # Basic checks
        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance.'
                            ''.format(domain))


        if ran is not None:
            # TODO
            raise NotImplementedError('custom range not implemented')
        else:
            ran = dom

        super().__init__(dom, ran, linear=True)

        # Handle kernel mode and impl
        impl = kwargs.pop('impl', 'default_ft')
        impl, impl_in = str(impl).lower(), impl
        if impl not in _CONV_SUPPORTED_IMPL:
            raise ValueError("implementation '{}' not understood."
                             ''.format(impl_in))
        self._impl = impl

        ker_mode = kwargs.pop('kernel_mode', 'real')
        ker_mode, ker_mode_in = str(ker_mode).lower(), ker_mode
        if ker_mode not in _CONV_SUPPORTED_KER_MODES:
            raise ValueError("kernel mode '{}' not understood."
                             ''.format(ker_mode_in))

        self._kernel_mode = ker_mode

        use_own_ft = (self.impl in ('default_ft', 'pyfftw_ft'))
        if not use_own_ft and self.kernel_mode != 'real':
            raise ValueError("kernel mode 'real' is required for impl "
                             "{}.".format(impl_in))

        self._axes = list(kwargs.pop('axes', (range(self.domain.ndim))))
        if ker_mode == 'real':
            halfcomplex = True  # efficient
        else:
            halfcomplex = ker_mode.endswith('hc')

        if use_own_ft:
            fft_impl = self.impl.split('_')[0]
            self._transform = FourierTransform(
                self.domain, axes=self.axes, halfcomplex=halfcomplex,
                impl=fft_impl)
        else:
            self._transform = None

        scale = kwargs.pop('scale', True)

        # TODO: handle case if axes are given, but the kernel is the same
        # for each point along the other axes
        if ker_mode == 'real':
            # Kernel given as real space element
            if use_own_ft:
                self._kernel = None
                self._kernel_transform = self.transform(kernel, **kwargs)
                if scale:
                    self._kernel_transform *= (np.sqrt(2 * np.pi) **
                                               self.domain.ndim)
            else:
                if not self.domain.partition.is_regular:
                    raise NotImplementedError(
                        'real-space convolution not implemented for '
                        'irregular sampling.')
                try:
                    # Function or other element-like as input. All axes
                    # must be used, otherwise we get an error.
                    # TODO: make a zero-centered space by default and
                    # adapt the range otherwise
                    self._kernel = self._kernel_elem(
                        self.domain.element(kernel).asarray(), self.axes)
                except (TypeError, ValueError):
                    # Got an array-like, axes can be used
                    self._kernel = self._kernel_elem(kernel, self.axes)
                finally:
                    self._kernel_transform = None
                    if scale:
                        self._kernel *= self.domain.cell_volume
        else:
            # Kernel given as Fourier space element
            self._kernel = None
            self._kernel_transform = self.transform.range.element(kernel)
            if scale:
                self._kernel_transform *= (np.sqrt(2 * np.pi) **
                                           self.domain.ndim)

    def _kernel_elem(self, kernel, axes):
        """Return kernel with adapted shape for real space convolution."""
        kernel = np.asarray(kernel)
        extra_dims = self.domain.ndim - kernel.ndim

        if extra_dims == 0:
            return self.domain.element(kernel)
        else:
            if len(axes) != extra_dims:
                raise ValueError('kernel dim {} + number of axes {} '
                                 '!= space dimension {}.'
                                 ''.format(kernel.ndim, len(axes),
                                           self.domain.ndim))

            # Sparse kernel (less dimensions), blow up
            slc = [None] * self.domain.ndim
            for ax in axes:
                slc[ax] = slice(None)

            kernel = kernel[slc]
            # Assuming uniform discretization
            min_corner = -self.domain.cell_sides * self.domain.shape / 2
            max_corner = self.domain.cell_sides * self.domain.shape / 2
            space = uniform_discr(min_corner, max_corner, kernel.shape,
                                  self.domain.exponent, self.domain.interp,
                                  impl=self.domain.impl,
                                  dtype=self.domain.dspace.dtype,
                                  order=self.domain.order,
                                  weighting=self.domain.weighting)
            return space.element(kernel)

    @property
    def impl(self):
        """Implementation of this operator."""
        return self._impl

    @property
    def kernel_mode(self):
        """The way in which the kernel is specified."""
        return self._kernel_mode

    @property
    def transform(self):
        """Fourier transform operator back-end if used, else `None`."""
        return self._transform

    @property
    def axes(self):
        """Axes along which the convolution is taken."""
        return self._axes

    @property
    def kernel(self):
        """Real-space kernel if used, else `None`.

        Note that this is the scaled version ``kernel * cell_volume``
        if ``scale=True`` was specified. Scaling here is more efficient
        than scaling the result.
        """
        return self._kernel

    @property
    def kernel_space(self):
        """Space of the convolution kernel."""
        return getattr(self.kernel, 'space', None)

    @property
    def kernel_transform(self):
        """Fourier-space kernel if used, else `None`.

        Note that this is the scaled version
        ``kernel_ft * (2*pi) ** (ndim/2)`` of the FT of the input
        kernel if ``scale=True`` was specified. Scaling here is more
        efficient than scaling the result.
        """
        return self._kernel_transform

    def _call(self, x, out, **kwargs):
        """Implement ``self(x, out[, **kwargs])``.

        Keyword arguments are passed on to the transform.
        """
        if self.kernel is not None:
            # Scipy based convolution
            out[:] = signal.convolve(x, self.kernel, mode='same')
        elif self.kernel_transform is not None:
            # Convolution based on our own transforms
            if self.domain.field == ComplexNumbers():
                # Use out as a temporary, has the same size
                # TODO: won't work for CUDA
                tmp = self.transform.range.element(out.asarray())
            else:
                # No temporary since out has reduced size (halfcomplex)
                tmp = None

            x_trafo = self.transform(x, out=tmp, **kwargs)
            x_trafo *= self.kernel_transform

            self.transform.inverse(x_trafo, out=out, **kwargs)
        else:
            raise RuntimeError('both kernel and kernel_transform are None.')

    def _adj_kernel(self, kernel, axes):
        """Return adjoint kernel with adapted shape."""
        kernel = np.asarray(kernel).conj()
        extra_dims = self.domain.ndim - kernel.ndim

        if extra_dims == 0:
            slc = [slice(None, None, -1)] * self.domain.ndim
            return kernel[slc]
        else:
            if len(axes) != extra_dims:
                raise ValueError('kernel dim ({}) + number of axes ({}) '
                                 'does not add up to the space dimension '
                                 '({}).'.format(kernel.ndim, len(axes),
                                                self.domain.ndim))

            # Sparse kernel (less dimensions), blow up
            slc = [None] * self.domain.ndim
            for ax in axes:
                slc[ax] = slice(None, None, -1)
            return kernel[slc]

    @property
    def adjoint(self):
        """Adjoint operator."""
        if self.kernel is not None:
            # TODO: this could be expensive. Move to init?
            adj_kernel = self._adj_kernel(self.kernel, self.axes)
            return Convolution(dom=self.domain,
                               kernel=adj_kernel, kernel_mode='real',
                               impl=self.impl, axes=self.axes)

        elif self.kernel_transform is not None:
            # TODO: this could be expensive. Move to init?
            adj_kernel_ft = self.kernel_transform.conj()

            if self.transform.halfcomplex:
                kernel_mode = 'ft_hc'
            else:
                kernel_mode = 'ft'
            return Convolution(dom=self.domain,
                               kernel=adj_kernel_ft, kernel_mode=kernel_mode,
                               impl=self.impl, axes=self.axes)
        else:
            raise RuntimeError('both kernel and kernel_transform are None.')


class RealSpaceConvolution(Convolution):

    """Convolution implemented in real space.

    This variant of the convolution is based on `scipy.signal.convolve`
    and is usually fast for small kernels. For a kernel of similar size
    as the input, `FourierSpaceConvolution` is probably much faster.
    """

    def __init__(self, domain, kernel, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Uniformly discretized space of functions on which the
            operator can act.
        kernel : `DiscreteLpVector`, callable or array-like
            Fixed kernel of the convolution operator. It can be
            specified in two ways, resulting in varying
            `Operator.range`. In both cases, the kernel must have
            the same number of dimensions as ``domain``, but the shapes
            can be different.

            `DiscreteLpVector` : The range is shifted by the midpoint
            of the kernel domain.

            callable or `array-like` : The kernel is interpreted as a
            continuous/discretized function with support centered around
            zero, which leads to the range of this operator being equal
            to its domain.

            See ``Notes`` for further explanation.

        scale : `bool`, optional
            If `True`, scale the discrete convolution with
            `DiscreteLp.cell_volume`, such that it approximates a
            continuous convolution.
            Default: `True`

        Other parameters
        ----------------
        resample : string or sequence of strings
            If ``kernel`` and ``domain`` have different cell sizes,
            this option defines the behavior of the evaluation. The
            following values can be given (per axis in the case of a
            sequence):

            'down' : Use the larger one of the two cell sizes (default).

            'up' : Use the smaller one of the two cell sizes. Note that
            this can be computationally expensive.

        kernel_scaled : bool, optional
            If True, the kernel is interpreted as already scaled
            as described under the ``scale`` argument.
            Default: False

        kernel_kwargs : dict, optional
            Keyword arguments passed to the call of the kernel function
            if given as a callable.

        See also
        --------
        scipy.signal.convolve : real-space convolution

        Examples
        --------
        >>> import odl
        >>> space = odl.uniform_discr(-5, 5, 5)
        >>> print(space.cell_sides)
        [ 2.]

        By default, the discretized convolution is scaled such that
        it approximates the convolution integral. This behavior can be
        switched off with ``scale=False``:

        >>> kernel = [-1, 1]
        >>> conv = RealSpaceConvolution(space, kernel)
        >>> conv([0, 1, 2, 0, 0])
        uniform_discr(-5.0, 5.0, 5).element([0.0, -2.0, -2.0, 4.0, 0.0])
        >>> conv_noscale = RealSpaceConvolution(space, kernel, scale=False)
        >>> conv_noscale([0, 1, 2, 0, 0])
        uniform_discr(-5.0, 5.0, 5).element([0.0, -1.0, -1.0, 2.0, 0.0])

        If the kernel is given as an element of a uniformly discretized
        function space, the convolution operator range is shifted by
        the midpoint of the kernel domain:

        >>> kernel_space = odl.uniform_discr(0, 4, 2)  # midpoint 2.0
        >>> kernel = kernel_space.element([-1, 1])
        >>> conv_shift = RealSpaceConvolution(space, kernel)
        >>> conv_shift.range  # Shifted by 2.0 in positive direction
        uniform_discr(-3.0, 7.0, 5)
        >>> conv_shift([0, 1, 2, 0, 0])
        uniform_discr(-3.0, 7.0, 5).element([0.0, -2.0, -2.0, 4.0, 0.0])

        Notes
        -----
        - The continuous convolution of two functions
          :math:`f` and :math:`g` defined on :math:`\mathbb{R^d}` is

              :math:`[f \\ast g](x) =
              \int_{\mathbb{R^d}} f(x - y) g(y) \mathrm{d}y`.

          This implementation covers the case of a fixed convolution
          kernel :math:`k`, i.e. the linear operator :math:`\mathcal{C}_k`
          given by

              :math:`\mathcal{C}_k(f): x \mapsto [k \\ast f](x)`.

        - If the convolution kernel :math:`k` is defined on a rectangular
          domain :math:`\Omega_0` with midpoint :math:`m_0`, then the
          convolution :math:`k \\ast f` of a function
          :math:`f:\Omega \\to \mathbb{R}` with :math:`k` has a support
          that is a superset of :math:`\Omega + m_0`. Therefore, we
          choose to keep the size of the domain and take
          :math:`\Omega + m_0` as the domain of definition of functions
          in the range of the convolution operator.

          In fact, the support of the convolution is contained in the
          `Minkowski sum
          <https://en.wikipedia.org/wiki/Minkowski_addition>`_
          :math:`\Omega + \Omega_0`.

          For example, if :math:`\Omega = [0, 5]` and
          :math:`\Omega_0 = [1, 3]`, i.e. :math:`m_0 = 2`, then
          :math:`\Omega + \Omega_0 = [1, 8]`. However, we choose the
          shifted interval :math:`\Omega + m_0 = [2, 7]` for the range
          of the convolution since it contains the "main mass" of the
          result.
        """
        # Basic checks
        if not isinstance(domain, DiscreteLp):
            raise TypeError('domain {!r} is not a DiscreteLp instance'
                            ''.format(domain))
        if not domain.is_uniform:
            raise ValueError('irregular sampling not supported')

        kernel_kwargs = kwargs.pop('kernel_kwargs', {})
        self._kernel, range = self._compute_kernel_and_range(
            kernel, domain, kernel_kwargs)

        super().__init__(domain, range, linear=True)

        # Hard-coding impl for now until we add more
        self._impl = 'scipy_convolve'

        # Handle the `resample` input parameter
        resample = kwargs.pop('resample', 'down')
        resample, resample_in = normalized_scalar_param_list(
            resample, length=self.domain.ndim,
            param_conv=lambda s: str(s).lower()), resample
        for i, (r, r_in) in enumerate(zip(resample, resample_in)):
            if r not in ('up', 'down'):
                raise ValueError("in axis {}: `resample` '{}' not understood"
                                 ''.format(i, r_in))
        self._resample = resample

        # Initialize resampling operators
        new_kernel_csides, new_domain_csides = self._get_resamp_csides(
            self.resample, self._kernel.space.cell_sides,
            self.domain.cell_sides)

        domain_resamp_space = uniform_discr_fromdiscr(
            self.domain, cell_sides=new_domain_csides)
        self._domain_resampling_op = Resampling(self.domain,
                                                domain_resamp_space)

        kernel_resamp_space = uniform_discr_fromdiscr(
            self._kernel.space, cell_sides=new_kernel_csides)
        self._kernel_resampling_op = Resampling(self._kernel.space,
                                                kernel_resamp_space)

        # Scale the kernel if desired
        kernel_scaled = kwargs.pop('kernel_scaled', False)
        self._kernel_scaling = np.prod(new_kernel_csides)
        self._scale = bool(kwargs.pop('scale', True))
        if self._scale and not kernel_scaled:
            # Don't modify the input
            self._kernel = self._kernel * self._kernel_scaling

    @staticmethod
    def _compute_kernel_and_range(kernel, domain, kernel_kwargs):
        """Return the kernel and the operator range based on the domain.

        This helper function computes an adequate zero-centered domain
        for the kernel if necessary (i.e. if the kernel is not a
        `DiscreteLpVector`) and calculates the operator range based
        on the operator domain and the kernel space.

        Parameters
        ----------
        kernel : `DiscreteLpVector` or array-like
            Kernel of the convolution.
        domain : `DiscreteLp`
            Domain of the convolution operator.
        kernel_kwargs : dict
            Used as ``space.element(kernel, **kernel_kwargs)`` if
            ``kernel`` is callable.

        Returns
        -------
        kernel : `DiscreteLpVector`
            Element in an adequate space.
        range : `DiscreteLp`
            Range of the convolution operator.
        """
        if isinstance(kernel, domain.element_type):
            if kernel.ndim != domain.ndim:
                raise ValueError('`kernel` has dimension {}, expected {}'
                                 ''.format(kernel.ndim, domain.ndim))
            # Set the range equal to the domain shifted by the midpoint of
            # the kernel space.
            kernel_shift = kernel.space.partition.midpoint
            range = uniform_discr_fromdiscr(
                domain, min_corner=domain.min_corner + kernel_shift)
            return kernel, range

        else:
            if callable(kernel):
                # Use "natural" kernel space, which is the zero-centered
                # version of the domain.
                extent = domain.partition.extent()
                std_kernel_space = uniform_discr_fromdiscr(
                    domain, min_corner=-extent / 2)
                kernel = std_kernel_space.element(kernel, **kernel_kwargs)
                range = domain
                return kernel, range
            else:
                # Make a zero-centered space with the same cell sides as
                # domain, but shape according to the kernel.
                kernel = np.asarray(kernel, dtype=domain.dtype)
                nsamples = kernel.shape
                extent = domain.cell_sides * nsamples
                kernel_space = uniform_discr_fromdiscr(
                    domain, min_corner=-extent / 2, max_corner=extent / 2,
                    nsamples=nsamples)
                kernel = kernel_space.element(kernel)
                range = domain
                return kernel, range

    @staticmethod
    def _get_resamp_csides(resample, ker_csides, dom_csides):
        """Return cell sides for kernel and domain resampling."""
        new_ker_csides = []
        new_dom_csides = []
        for resamp, ks, ds in zip(resample, ker_csides, dom_csides):

            if np.isclose(ks, ds):
                # Keep old if both csides are the same
                new_ker_csides.append(ks)
                new_dom_csides.append(ds)
            else:
                # Pick the coarser one if 'down' or the finer one if 'up'
                if ((ks < ds and resamp == 'up') or
                        (ks > ds and resamp == 'down')):
                    new_ker_csides.append(ks)
                    new_dom_csides.append(ks)
                else:
                    new_ker_csides.append(ds)
                    new_dom_csides.append(ds)

        return new_ker_csides, new_dom_csides

    @property
    def impl(self):
        """Implementation of this operator."""
        return self._impl

    @property
    def resample(self):
        """Resampling used during evaluation."""
        return self._resample

    def kernel(self):
        """Return the real-space kernel of this convolution operator.

        Note that internally, the scaled kernel is stored if
        ``scale=True`` (default) was chosen during initialization, i.e.
        calculating the original kernel requires to reverse the scaling.
        Otherwise, the original unscaled kernel is returned.
        """
        if self._scale:
            return self._kernel / self._kernel_scaling
        else:
            return self._kernel

    def _call(self, x):
        """Implement ``self(x)``."""
        if np.allclose(self._kernel.space.cell_sides, self.domain.cell_sides):
            # No resampling needed
            return signal.convolve(x, self._kernel, mode='same')
        else:
            if self.domain != self._domain_resampling_op.range:
                # Only resample if necessary
                x = self._domain_resampling_op(x)

            if self._kernel.space != self._kernel_resampling_op.range:
                kernel = self._kernel_resampling_op(self._kernel)
            else:
                kernel = self._kernel

            conv = signal.convolve(x, kernel, mode='same')
            return self._domain_resampling_op.inverse(conv)

    @property
    def adjoint(self):
        """Adjoint operator defined by the reversed kernel."""
        # Make space for the adjoint kernel, -S if S is the kernel space
        adj_min_corner = -self._kernel.space.max_corner
        adj_max_corner = -self._kernel.space.min_corner
        adj_ker_space = uniform_discr_fromdiscr(
            self._kernel.space,
            min_corner=adj_min_corner, max_corner=adj_max_corner)

        # Adjoint kernel is k_adj(x) = k(-x), i.e. the kernel array is
        # reversed in each axis.
        reverse_slice = (slice(None, None, -1),) * self.domain.ndim
        adj_kernel = adj_ker_space.element(
            self._kernel.asarray()[reverse_slice])

        return RealSpaceConvolution(self.range, adj_kernel, impl=self.impl,
                                    resample=self.resample, scale=True,
                                    kernel_scaled=True)


def resize_array_symm(arr, newshp, out=None):
    """Return the resized version of ``arr`` with shape ``newshp``.

    In axes where ``newshp > arr.shape``, zeros are added evenly to
    the left and the right, with preference for right if the number
    of added zeros is odd.
    Where ``newshp < arr.shape``, entries are discarded, in the same
    symmetric manner as above.

    Parameters
    ----------
    arr : array-like
        Array to be resized.
    newshp : int or sequence of int
        Desired shape of the output array.
    out : `numpy.ndarray`, optional
        Array to write the result to. Must have shape ``newshp`` and
        be able to hold the data type of the input array.

    Returns
    -------
    resized : `numpy.ndarray`
        Resized array created according to the above rules. If ``out``
        was given, the returned object is a reference to it.
    """
    arr = np.asarray(arr)
    newshp = tuple(normalized_scalar_param_list(newshp, arr.ndim,
                                                param_conv=safe_int_conv))
    if out is None:
        out = np.zeros(newshp, dtype=arr.dtype)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('`out` must be a `numpy.ndarray` instance, got '
                            '{!r}'.format(out))
        if not np.can_cast(arr.dtype, out.dtype):
            raise ValueError('data type {} of `arr` cannot be safely cast '
                             'to data type {} of `out`'
                             ''.format(arr.dtype, out.dtype))
        if out.shape != newshp:
            raise ValueError('`out` must have shape {}, got {}'
                             ''.format(newshp, out.shape))
        out.fill(0.0)

    arr_slc, out_slc = [], []
    for n_old, n_new in zip(arr.shape, out.shape):
        if n_new > n_old:
            iremove = n_new - n_old
            istart = iremove // 2
            istop = -(iremove - istart)
            arr_slc.append(slice(None))
            out_slc.append(slice(istart, istop))
        elif n_new < n_old:
            iadd = n_old - n_new
            istart = iadd // 2
            istop = -(iadd - istart)
            arr_slc.append(slice(istart, istop))
            out_slc.append(slice(None))
        else:
            arr_slc.append(slice(None))
            out_slc.append(slice(None))

    arr_slc = tuple(arr_slc)
    out_slc = tuple(out_slc)
    out[out_slc] = arr[arr_slc]
    return out


def zero_centered_discr_fromdiscr(discr, shape=None):
    """Return a discretization centered around zero.

    The cell sizes will be kept in any case, but the total size can
    be changed with the ``shape`` option.

    Parameters
    ----------
    discr : `DiscreteLp`
        Uniformly discretized space to be used as template.
    shape : int or sequence of int, optional
        If specified, resize the space to this shape.

    Returns
    -------
    newspace : `DiscreteLp`
        Uniformly discretized space whose domain has midpoint
        ``(0, ..., 0)`` and the same cell size as ``discr``.
    """
    if not isinstance(discr, DiscreteLp):
        raise TypeError('`discr` {!r} is not a DiscreteLp instance'
                        ''.format(discr))
    if not discr.is_uniform:
        raise ValueError('irregular sampling not supported')

    if shape is not None:
        shape = normalized_scalar_param_list(shape, discr.ndim,
                                             param_conv=safe_int_conv)
    else:
        shape = discr.shape
    new_min_corner = -(discr.cell_sides * shape) / 2
    return uniform_discr_fromdiscr(discr, min_corner=new_min_corner,
                                   nsamples=shape)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
