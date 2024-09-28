# ode-glucose-forecasting
The classes MultivariateForecastingEnvironment and ForecastingUtils are used to pre-process a dataset in an online manner such that the ODE states are available for use in a CSV format for the forecast script to run on.

to run the forecasting script run
`python -m ode.forecast --key {folderName}`

It is expected to have folders following the regex pattern in the Participant class, where different csv files are for univariate, multivariate (with ode states) and two variants, acc and acc_rand for the 75% and 50% accurate estimations of the meals.

## Pre-requisites
You need the following repository downloaded and installed in your environment to use the `MultivariateForecastingEnvironment` class or the `estimateMeals` script to generate the state ODE's https://github.com/DaneLyttinen/T2DM-sim
