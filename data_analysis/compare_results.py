#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage: compare_results.py <result_a> <result_b>
"""

from docopt import docopt

import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":


	args = docopt(__doc__)

	ra = pd.read_csv(args["<result_a>"])
	rb = pd.read_csv(args["<result_b>"])

	results = pd.merge(ra, rb, on="week")
	mse_a = mean_squared_error(results["incidence_x"], results["prediction_x"])
	mse_b = mean_squared_error(results["incidence_y"], results["prediction_y"])

	improvement_rate = mse_a - mse_b


	print("MSE file A: {}".format(mse_a))
	print("MSE file B: {}".format(mse_b))
	print("Improvement Rate A-B: {}".format(improvement_rate))
