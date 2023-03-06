# Literature Survey:

1. [Data-driven discovery of governing equations for coarse-grained
heterogeneous network dynamics](https://arxiv.org/pdf/2205.10965.pdf).  
    - Proper orthogonal decomposition (POD) 1,6
    - Dynamic node decomposition (DMD) 7,8 - both approximate linear subspaces using dominant correlations in spatio-temporal data.  highly restrictive and ill-suited to
handle parametric dependencies
    - multiple linear subspaces covering different temporal or spatial domains [9, 10, 11, 12]
    -  multi-resolution DMD [13]
    - deep learning to compute underlying nonlinear subspaces which are advantageous for dynamics, both linear and nonlinear [18, 19, 20, 21, 22]
    - Gottwald originally introduced collective coordinates to describe coupled phase oscillators of the Kuramoto model [25], and then extended the mathematical framework to reduce infinite dimensional stochastic partial differential equations (SPDEs) with symmetry to a set of finite-dimensional stochastic differential equations which describe the shape of the solution and the dynamics along the symmetry group [26]
    -  Sparse Identification of Nonlinear Dynamics (SINDy) with trimming
    - In this work, we leverage dimensionality-reduction, sparse regression, and robust statistics to discover coarse-grained models of heterogeneous networked dynamical systems.
    - the seminal review of Cross and Hohenberg [23] detail the many mathematical architectures available for deriving order-parameter descriptions, or coarse-grained models, that characterize the dynamics.
    - statistical and machine learning algorithms [36, 37] for network clustering
    - Assume nonlinear dynamic system dx/dt = f(x,t)
    -  The method introduced in [18] seeks to identify f via sequential threshold least-squares
