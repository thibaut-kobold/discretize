from __future__ import division
import numpy as np
import scipy.sparse as sp

from matrixutils import (
    mkvc, sdiag, sdInv, speye, kron3, spzeros, ddx, av, av_extrap, ndgrid,
    ind2sub, sub2ind, getSubArray, inv3X3BlockDiagonal, inv2X2BlockDiagonal,
    Zero, Identity, isScalar
)

class TensorType(object):
    def __init__(self, M, tensor):
        if tensor is None:  # default is ones
            self._tt = -1
            self._tts = 'none'
        elif isScalar(tensor):
            self._tt = 0
            self._tts = 'scalar'
        elif tensor.size == M.nC:
            self._tt = 1
            self._tts = 'isotropic'
        elif (
            (M.dim == 2 and tensor.size == M.nC*2) or
            (M.dim == 3 and tensor.size == M.nC*3)
        ):
            self._tt = 2
            self._tts = 'anisotropic'
        elif (
            (M.dim == 2 and tensor.size == M.nC*3) or
            (M.dim == 3 and tensor.size == M.nC*6)
        ):
            self._tt = 3
            self._tts = 'tensor'
        else:
            raise Exception(
                'Unexpected shape of tensor: {}'.format(tensor.shape)
            )

    def __str__(self):
        return 'TensorType[{0:d}]: {1!s}'.format(self._tt, self._tts)

    def __eq__(self, v):
        return self._tt == v

    def __le__(self, v):
        return self._tt <= v

    def __ge__(self, v):
        return self._tt >= v

    def __lt__(self, v):
        return self._tt < v

    def __gt__(self, v):
        return self._tt > v


def makePropertyTensor(M, tensor):
    if tensor is None:  # default is ones
        tensor = np.ones(M.nC)

    if isScalar(tensor):
        tensor = tensor * np.ones(M.nC)

    propType = TensorType(M, tensor)
    if propType == 1:  # Isotropic!
        Sigma = sp.kron(sp.identity(M.dim), sdiag(mkvc(tensor)))
    elif propType == 2:  # Diagonal tensor
        Sigma = sdiag(mkvc(tensor))
    elif M.dim == 2 and tensor.size == M.nC*3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((M.nC, 3), order='F')
        row1 = sp.hstack((sdiag(tensor[:, 0]), sdiag(tensor[:, 2])))
        row2 = sp.hstack((sdiag(tensor[:, 2]), sdiag(tensor[:, 1])))
        Sigma = sp.vstack((row1, row2))
    elif M.dim == 3 and tensor.size == M.nC*6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((M.nC, 6), order='F')
        row1 = sp.hstack(
            (sdiag(tensor[:, 0]), sdiag(tensor[:, 3]), sdiag(tensor[:, 4]))
        )
        row2 = sp.hstack(
            (sdiag(tensor[:, 3]), sdiag(tensor[:, 1]), sdiag(tensor[:, 5]))
        )
        row3 = sp.hstack(
            (sdiag(tensor[:, 4]), sdiag(tensor[:, 5]), sdiag(tensor[:, 2]))
        )
        Sigma = sp.vstack((row1, row2, row3))
    else:
        raise Exception('Unexpected shape of tensor')

    return Sigma


def invPropertyTensor(M, tensor, returnMatrix=False):

    propType = TensorType(M, tensor)

    if isScalar(tensor):
        T = 1./tensor
    elif propType < 3:  # Isotropic or Diagonal
        T = 1./mkvc(tensor)  # ensure it is a vector.
    elif M.dim == 2 and tensor.size == M.nC*3:  # Fully anisotropic, 2D
        tensor = tensor.reshape((M.nC, 3), order='F')
        B = inv2X2BlockDiagonal(tensor[:, 0], tensor[:, 2],
                                tensor[:, 2], tensor[:, 1],
                                returnMatrix=False)
        b11, b12, b21, b22 = B
        T = np.r_[b11, b22, b12]
    elif M.dim == 3 and tensor.size == M.nC*6:  # Fully anisotropic, 3D
        tensor = tensor.reshape((M.nC, 6), order='F')
        B = inv3X3BlockDiagonal(tensor[:, 0], tensor[:, 3], tensor[:, 4],
                                tensor[:, 3], tensor[:, 1], tensor[:, 5],
                                tensor[:, 4], tensor[:, 5], tensor[:, 2],
                                returnMatrix=False)
        b11, b12, b13, b21, b22, b23, b31, b32, b33 = B
        T = np.r_[b11, b22, b33, b12, b13, b23]
    else:
        raise Exception('Unexpected shape of tensor')

    if returnMatrix:
        return makePropertyTensor(M, T)

    return T


