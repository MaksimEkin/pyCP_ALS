"""
CP-ALS tensor decomposition.

References
========================================
[1] General software, latest release: Brett W. Bader, Tamara G. Kolda and others, Tensor Toolbox for MATLAB, Version 3.2.1, www.tensortoolbox.org, April 5, 2021.

[2] Dense tensors: B. W. Bader and T. G. Kolda, Algorithm 862: MATLAB Tensor Classes for Fast Algorithm Prototyping, ACM Trans. Mathematical Software, 32(4):635-653, 2006, http://dx.doi.org/10.1145/1186785.1186794.

[3] Sparse, Kruskal, and Tucker tensors: B. W. Bader and T. G. Kolda, Efficient MATLAB Computations with Sparse and Factored Tensors, SIAM J. Scientific Computing, 30(1):205-231, 2007, http://dx.doi.org/10.1137/060676489.

"""
# -*- coding: utf-8 -*-
from .ktensor import K_TENSOR
from .sptensor import SP_TENSOR
from .khatrirao_sptensor import khatrirao as mttkrp
from .norm_ktensor import norm
from .arrange_ktensor import arrange
from .fixsigns_ktensor import fixsigns_oneargin
from .innerprod_ktensor import innerprod

from tqdm import tqdm
import numpy as np


class CP_ALS:
    def __init__(
        self, tol=1e-4, n_iters=50, verbose=True, fixsigns=True, random_state=42
    ):
        """
        Initilize the CP-ALS object.
        
        Parameters
        ----------
        tol : float, optional
            Tolerance on difference in fit. Default is 1e-4.
        n_iters : int, optional
            Number of iterations. Defailt is 50.
        verbose : bool, optional
            If True, shows progress. Defailt is True.
        fixsigns : bool, optional
            If True, call fixsigns at end of iterations. Defailt is True.  
        random_state : int, optional
            Random seed. Defailt is 42.  

        """

        self.fitchangetol = tol
        self.n_iters = n_iters
        self.verbose = verbose
        self.fixsigns = fixsigns
        self.random_state = random_state
        self.dimorder = []
        np.random.seed(self.random_state)

    def fit(self, coords=[], values=[], rank=2, Minit="random"):
        """
        Takes the decomposition of sparse tensor X and returns the KRUSKAL tensor M.\n
        Here M is  latent factors and the weight of each R (rank) component.\n
        
        Parameters
        ----------
        coords : Numpy array
            Array of non-zero coordinates for sparse tensor X. COO format.\n
            Each entry in this array is a coordinate of a non-zero value in the original tensor X.
            .. warning::
            
                * ``len(coords)`` is number of total entiries in X, and ``len(coords[0])`` should give the number of dimensions X has.
                
        values : Numpy array
            List of non-zero values corresponding to each list of non-zero coordinates (``coords``).
            Array of non-zero tensor entries. COO format.
            .. warning::
                * Length of ``values`` must match the length of ``coords``.
                
        rank : int
            Tensor rank.\n
            Tensor rank determines the number of components.\n
            The default is ``rank=2``.
            
        Minit : string or dictionary of latent factors
            Initial value of latent factors.\n
            If ``Minit='random'``, initial factors are drawn randomly from uniform distribution between 0 and 1.\n
            Else, pass a dictionary where the key is the mode number and value is array size ``d x r``
            where ``d`` is the number of elements on the dimension and ``r`` is the rank.\n
            The default is ``Minit='random'``.
            
            .. note::
            
                Example on creating initial M for 3 dimensional tensor shaped *5x5x5* for rank 4 decomposition:
                .. code-block:: python
                    import numpy as np
                    num_dimensions = 3
                    tensor_shape = [5,5,5]
                    rank = 4
                    M_init = {"Factors":{}, "Weights":[1,1,1]}
                    for d in range(num_dimensions):
                            M_init["Factors"][str(d)] = np.random.uniform(low=0, high=1, size=(tensor_shape[d], rank))
                    M_init["Factors"]
                    
                .. code-block:: console
                
                    {
                     '0': array([[0.821161  , 0.419537  , 0.62692165, 0.06294969],
                            [0.02032657, 0.88625546, 0.74128504, 0.71855629],
                            [0.70760879, 0.83813636, 0.35128158, 0.94442011],
                            [0.35780608, 0.83703369, 0.84602297, 0.93760842],
                            [0.00746915, 0.05974905, 0.49097518, 0.60615737]]),
                     '1': array([[0.61902526, 0.78453503, 0.05596952, 0.69149084],
                            [0.56300552, 0.82418509, 0.04278352, 0.25716303],
                            [0.66221183, 0.13888761, 0.92502242, 0.57817265],
                            [0.31738958, 0.87061048, 0.64170398, 0.62236073],
                            [0.9110603 , 0.5133135 , 0.89232955, 0.09881775]]),
                     '2': array([[0.0580065 , 0.82367217, 0.07616138, 0.93873983],
                            [0.89247679, 0.41388867, 0.82089524, 0.10293565],
                            [0.13540868, 0.09809637, 0.10844113, 0.90405324],
                            [0.91167498, 0.67068632, 0.51705956, 0.82211517],
                            [0.80942828, 0.08450466, 0.6306868 , 0.78132797]])
                    }

        Returns
        -------
        KRUSKAL tensor M : dict
            KRUSKAL tensor M is returned in dict format.\n
            The latent factors can be found with the key 'Factors'.\n
            The weight of each component can be found with the key 'Weights'.
        .. note::
        """
        
        if rank <= 0:
            raise Exception("Number of components requested must be positive")

        #
        #  Set up for iterations - initializing M and the fit.
        #
        X, M = self.__setup(coords, values, Minit, rank)

        # Extract number of dimensions and norm of X.
        N = X.Dimensions
        normX = np.linalg.norm(X.data, ord=2)

        self.dimorder = np.arange(0, N)

        fit = 0
        R = rank
        M_mttkrp = np.zeros((X.Size[-1], R))

        if self.verbose:
            print("CP_ALS:")

        #
        # Main Loop: Iterate until convergence
        #
        UtU = np.zeros((R, R, N))
        for n in range(N):
            if len(M.Factors[str(n)]) != 0:
                UtU[:, :, n] = np.dot(M.Factors[str(n)].T, M.Factors[str(n)])

        for itr in tqdm(range(self.n_iters), disable=not (self.verbose)):

            fitold = fit

            #
            # Iterate over all N modes of the tensor
            #
            for n in self.dimorder:

                # Calculate Unew = X_(n) * khatrirao(all M except n, 'r').
                Unew = mttkrp(X, M, n)

                # Save the last MTTKRP result for fitness check.
                if n == self.dimorder[-1]:
                    U_mttkrp = Unew

                # Compute the matrix of coefficients for linear system
                target_dimensions = list(np.arange(0, N))
                target_dimensions.pop(target_dimensions.index(n))
                Y = np.prod(UtU[:, :, target_dimensions], 2)
                Unew = np.linalg.lstsq(Y.T, Unew.T, rcond=None)[0].T

                # Normalize each vector to prevent singularities in coefmatrix
                if itr == 0:
                    lambda_ = np.sqrt(np.sum(Unew ** 2, axis=0)).T  # 2-norm
                else:
                    lambda_ = np.max(np.abs(Unew), axis=0).T  # max-norm

                Unew = np.divide(Unew, lambda_)

                M.Factors[str(n)] = Unew
                UtU[:, :, n] = np.dot(M.Factors[str(n)].T, M.Factors[str(n)])

            Utmp = {"Factors": [], "Weights": []}
            Utmp["Factors"] = M.deep_copy_factors()
            Utmp["Weights"] = lambda_
            P = K_TENSOR(Rank=R, Size=M.Size, Minit=Utmp)

            # This is equivalent to innerprod(X,P).
            iprod = np.sum(
                np.multiply(
                    np.sum(
                        np.multiply(M.Factors[str(self.dimorder[-1])], U_mttkrp), axis=0
                    ),
                    lambda_.T,
                )
            )

            if normX == 0:
                fit = norm(P) ** 2 - 2 * iprod
            else:
                normresidual = np.sqrt(normX ** 2 + norm(P) ** 2 - 2 * iprod)
                fit = 1 - (normresidual / normX)

            fitchange = np.abs(fitold - fit)

            # Check for convergence
            if (itr > 0) and (fitchange < self.fitchangetol):
                converged = True
            else:
                converged = False

            if converged:
                break

        #
        # Clean up final result
        #
        P = arrange(P)
        if self.fixsigns:
            P = fixsigns_oneargin(P)

        if self.verbose:

            if normX == 0:
                fit = norm(P) ** 2 - 2 * innerprod(P, X)
            else:
                normresidual = np.sqrt(normX ** 2 + norm(P) ** 2 - 2 * innerprod(P, X))
                fit = 1 - (normresidual / normX)

            print("\nFinal fit=", fit)

        results = {"Factors": [], "Weights": []}
        results["Factors"] = P.Factors
        results["Weights"] = P.Weights

        return results

    def __setup(self, coords, values, Minit, rank):

        if len(coords) == 0:
            raise Exception(
                "Coordinates of the non-zero elements is not passed for sptensor.\
                                Use the Coords parameter."
            )
        if len(values) == 0:
            raise Exception(
                "Non-zero values are not passed for sptensor.\
                            Use the Values parameter"
            )
        if (values < 0).all():
            raise Exception(
                "Data tensor must be nonnegative for Poisson-based factorization"
            )

        X = SP_TENSOR(coords, values)

        M = K_TENSOR(rank, X.Size, Minit, self.random_state)

        return X, M
