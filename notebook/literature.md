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


Deep learning in Dynamcic systems
[Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/pdf/2102.11174.pdf).  [Video discussion](https://www.youtube.com/watch?v=RSSVWpBak6s).  Makes a strong parallel between transformers and LTI systems treated as a data retrieval problem.  Investigates eliminating the softmax.  

[Tensorized Transformer for Dynamical Systems Modeling](https://arxiv.org/pdf/2006.03445.pdf)
[Transformers for modeling physical systems](https://openreview.net/pdf/c45d1ade1683075a8a4e5bfe568cf3915805af44.pdf).  [Related works](https://deepai.org/publication/tensorized-transformer-for-dynamical-systems-modeling)
[Deep Learning for Dynamics — the Intuitions](https://towardsdatascience.com/deep-learning-for-dynamics-the-intuitions-20a67942dfbc)
[DNN - Tutorial 2 Part I: Physics inspired Machine Learning](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_1.html)
[Integrating Deep Learning with the Theory of Nonlinear, Chaotic, and History-Dependent Dynamics](https://dataspace.princeton.edu/handle/88435/dsp01r494vp26b)
[Reconstruction of the flame nonlinear response using deep learning algorithms](https://aip.scitation.org/doi/10.1063/5.0131928)
[Redesigning the Transformer Architecture with Insights from Multi-particle Dynamical Systems](https://proceedings.neurips.cc/paper/2021/file/2bd388f731f26312bfc0fe30da009595-Paper.pdf)
[Deep Learning to Discover Coordinates for Dynamics: Autoencoders & Physics Informed Machine Learning](https://www.youtube.com/watch?v=KmQkDgu-Qp0)
[Sparse Identification of Nonlinear Dynamics (SINDy): Sparse Machine Learning Models 5 Years Later!](https://www.youtube.com/watch?v=NxAn0oglMVw)
[Bridging Physics-based and Data-driven modeling for Learning Dynamical Systems](https://par.nsf.gov/servlets/purl/10322792)