# PY OP
##### Py Op is a options research infrastructure meant to assist in option trading and research. It uses design patterns and systemn design principles to create robust scalable code where the focus is to minimize coupling between part and modules of the project.
##### There are three main modules in this library calc_engine, analysis and data
### 1. Calc_engine:
#### Calc_engine summary
##### The calculation engine is where all the models are run in the library and where all calculations are done. This module does not load, store, ingest or even touch the data, the data is passed into classes and functions in this module. So far the engine has functionality for option pricing, stochastic volatility model calibration, volatility surface interpolation, greek calculation and implied density extraction.
#### Calc_engine Option Pricer
##### Has the functionality for European option pricing using Black-Scholes, Bachelier, SABR, Heston, Merton-Jump, Variance-Gamma and rBergomi. Can price using FFT, Fourier inversion, Monte Carlo simulation, Analytical formulas and quadrature techniques. 
##### Can price American Options using Least Squares Monte Carlo, Longstaff-Schwartz, Cox-Ross-Rubenstein Binomial Tree and PDE finite difference methods.
##### Calc_engine option price calibration can calibrate Heston, SABR and rBergomi parameters.
#### Calc_engine Density Extraction
##### So far we have functionality for Breeded-Litzenberger or Weighted Monte Carlo density extraction
#### Vol_engine summary
##### This is the portion of calc_engine that handels everything with volatility, we have multiple classes that calculate implied volatility (Newton, Secant, Bisection, Root Finder), and we have a class to handle the retrieval of volaitltiy skew and term structure.
##### The interpolation file uses methodologies for fitting the implied volatility skew such as SVI, Polynomial, GVV, GVV+, GVV5
