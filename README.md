# bayes_drt
`bayes_drt` is a Python package for inverting  electrochemical impedance spectroscopy (EIS) data to obtain the distribution of relaxation times (DRT) and/or distribution of diffusion times (DDT).

`bayes_drt` implements a hierarchical Bayesian model to provide well-calibrated estimates of the DRT or DDT without ad-hoc tuning. The package offers two methods for solving the model:
* Hamiltonian Monte Carlo (HMC) sampling to estimate the posterior distribution, providing both a point estimate of the distribution and a credible interval
* L-BFGS optimization maximize the posterior probability, providing a maximum *a posteriori* (MAP) point estimate of the distribution
It is also possible to perform multi-distribution inversions, e.g. to simultaneously fit both a DRT and a DDT, with these methods. This is an experimental feature and requires some manual tuning. See the tutorials for an example.

The package also provides ordinary and hyperparametric ridge regression methods, which may be useful for comparison or for obtaining initial estimates of the distribution. The hyperparametric ridge regression method is an implementation of the method developed by Ciucci and Chen (https://doi.org/10.1016/j.electacta.2015.03.123) and expanded by Effat and Ciucci (https://doi.org/10.1016/j.electacta.2017.07.050).

Several tutorials are available in `tutorials`. Additional examples and documentation will be added soon.

The methods implemented in `bayes_drt` are the subject of a manuscript submitted to *Electrochimica Acta* in Aug. 2020. All code used to generate the results in the manuscript are available here:
* `data` contains all experimental and simulated data files.
* `code_EchemActa` contains the code used to simulate data, estimate the DRT and DDT using `bayes_drt`, apply several other inversion methods from the literature to the same data for comparison, and generate figures.

## Installation



## Citing `bayes_drt`

If you use `bayes_drt` to obtain results which are published in or used for an academic journal article, please cite the following:
* Huang, J., Papac, M., and O'Hayre, R. (2020). Towards robust autonomous impedance spectroscopy analysis: a calibrated hierarchical Bayesian approach for electrochemical impedance spectroscopy (EIS) inversion. Electrochimica Acta, *in press*.

Additionally, if you use the `ridge_fit` method with `hyper_lambda=True` or `hyper-w=True`, please cite the corresponding work below:
* `hyper_lambda=True`: Ciucci, F., & Chen, C. (2015). Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach. Electrochimica Acta, 167, 439–454. https://doi.org/10.1016/j.electacta.2015.03.123
* `hyper_w=True`: Effat, M. B., & Ciucci, F. (2017). Bayesian and Hierarchical Bayesian Based Regularization for Deconvolving the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data. Electrochimica Acta, 247, 1117–1129. https://doi.org/10.1016/J.ELECTACTA.2017.07.050
