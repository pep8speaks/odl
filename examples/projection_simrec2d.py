# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of ODL.

ODL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ODL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ODL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
from future import standard_library
from math import sin, cos
import matplotlib.pyplot as plt

import numpy as np
import odl.operator.operator as OP
import odl.space.function as fs
import odl.space.cartesian as ds
import odl.discr.discretization as dd
import odl.space.set as sets
import SimRec2DPy as SR

standard_library.install_aliases()


class Projection(OP.LinearOperator):
    def __init__(self, volumeOrigin, voxelSize, volumeSize, detectorSize,
                 stepSize, sourcePosition, detectorOrigin, pixelDirection,
                 domain, range_):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection
        self.domain = domain
        self.range = range_

    def _apply(self, data, out):
        forward = SR.SRPyForwardProject.SimpleForwardProjector(
            data.data.reshape(self.volumeSize), self.volumeOrigin,
            self.voxelSize, self.detectorSize, self.stepSize)

        result = forward.project(self.sourcePosition, self.detectorOrigin,
                                 self.pixelDirection)

        out[:] = result.transpose()

# Set geometry parameters
volumeSize = np.array([20.0, 20.0])
volumeOrigin = -volumeSize/2.0

detectorSize = 50.0
detectorOrigin = -detectorSize/2.0

sourceAxisDistance = 20.0
detectorAxisDistance = 20.0

# Discretization parameters
nVoxels = np.array([500, 400])
nPixels = 400
nProjection = 500

# Scale factors
voxelSize = volumeSize/nVoxels
pixelSize = detectorSize/nPixels
stepSize = voxelSize.max()/2.0

# Define projection geometries
theta = 0
x0 = np.array([cos(theta), sin(theta)])
y0 = np.array([-sin(theta), cos(theta)])

sourcePosition = -sourceAxisDistance * x0
detectorOrigin = detectorAxisDistance * x0 + detectorOrigin * y0
pixelDirection = y0 * pixelSize


dataSpace = fs.L2(sets.Interval(0, 1))
dataRn = ds.En(nPixels)
dataDisc = dd.uniform_discretization(dataSpace, dataRn)

reconSpace = fs.L2(sets.Rectangle((0, 0), (1, 1)))
reconRn = ds.En(nVoxels.prod())
reconDisc = dd.uniform_discretization(reconSpace, reconRn, nVoxels)

# Create a phantom
phantom = SR.SRPyUtils.phantom(nVoxels)
phantomVec = reconDisc.element(phantom)

projector = Projection(volumeOrigin, voxelSize, nVoxels, nPixels, stepSize,
                       sourcePosition, detectorOrigin, pixelDirection,
                       reconDisc, dataDisc)

result = dataDisc.element()
projector.apply(phantomVec, result)

plt.plot(result[:])
plt.show()
