#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage: compare_results.py <result_a> <result_b> [--f]
"""

from docopt import docopt

import pandas as pd

if __name__ == "__main__":


	args = docopt(__doc__)

	ra = pd.read_csv(args["<result_a>"])
	rb = pd.read_csv(args["<result_b>"])

	results = pd.merge(ra, rb, on="week")
	results["diff_x"] = pow(results["incidence_x"]-results["prediction_x"], 2)
	results["diff_y"] = pow(results["incidence_y"]-results["prediction_y"], 2)
	results["improvement_rate"] = results["diff_x"]-results["diff_y"]

	print(results["improvement_rate"])
