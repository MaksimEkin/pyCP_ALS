#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tenmat.py creates a matricized tensor.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.

[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.

@author: Maksim Ekin Eren
"""
import copy
import numpy as np
from .permute_ktensor import permute
from .double_ktensor import double


def tenmat(X, mode):
    """
    Create a matricized tensor.
    Parameters
    ----------
    X : class
        Kruskal tensor, ktensor.K_TENSOR.
    mode : int
        Dimension number to unfold on.

    Returns
    -------
    X : np.ndarray
        Matriced version of the sparse tensor in as dense matrix.

    """
    rdims = [mode]
    tmp = [True] * len(X.Size)
    tmp[rdims[0]] = False
    cdims = np.where(tmp)[0]
    order = rdims + list(cdims)

    X_t = permute(copy.deepcopy(X), order)

    x = np.prod([X.Size[i] for i in rdims])
    y = np.prod([X.Size[i] for i in cdims])

    A = double(X_t)
    return np.reshape(A, [x, y])
