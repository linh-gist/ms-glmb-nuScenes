IMPLEMENTATION
========================
- This Python implementation is ported from an implemented version of multi-sensor delta-GLMB, The Unscented Kalman Filter for Nonlinear Estimation.
- Sampling solutions (ranked assignments), `gibbs_multisensor_approx_cheap` is implemented in C++ based on __Algorithm 2: MM-Gibbs (Suboptimal)__ [0]. 
- A version of MS-GLMB C++ is also implemented (check out `/cpp_gibbs/src/ms_glmb_ukf`.

[0] Vo, B. N., Vo, B. T., & Beard, M. (2019). Multi-sensor multi-object tracking with the generalized labeled multi-Bernoulli filter. IEEE Transactions on Signal Processing, 67(23), 5952-5967.  

USAGE
=====
- Install packages.
    - `numpy`
    - `scipy`
    - `h5py` (read camera matrix from Matlab matrix)
    - `matplotlib(3.4.1)` draw result
    - C++ packages (`gibbs_multisensor_approx_cheap`, `ms_glmb_ukf`) in cpp_gibbs (`python setup.py build develop`).  
- Run: `python demo.py`.
- Use the following statements to run `python demo.py` with C++ (Note: make sure to comment Python code).

            glmb.runcpp(model_params, dataset, meas)

LICENCE
=======
Linh Ma (`linh.mavan@gm.gist.ac.kr`), Machine Learning & Vision Laboratory, GIST, South Korea

