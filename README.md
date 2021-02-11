# Predicting median housing prices in Boston

## Objective
The objective is to predict median home prices in Boston based on data published in 1978.  

The data was provided by U.S. Census service and obtained from the [StatLib Archive](http://lib.stat.cmu.edu/datasets/boston).  It has been used extensively throughout the literature to benchmark algorithms. The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978. 

</br>

## Executive summary

The executive summary is located in [REPORT.md](./REPORT.md)

<br>

## Usage Instructions

1. Clone the repository: `git clone https://github.com/bxp151/housing.git
`
2. Install the required packages: `pip install -r requirements.txt `
3. Open `housing.py` and set `HOME_DIR` to the directory path where you cloned the repository
4. In `housing.py`, I commented out the `Execute target shuffling` code block starting on `line 453` because it takes several hours to run.  The results of the execution will be loaded automatically into the gb_scores variable.
5. Run `housing.py` to reproduce the results

</br>

