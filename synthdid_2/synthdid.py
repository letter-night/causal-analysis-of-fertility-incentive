import numpy as np, pandas as pd
import itertools, matplotlib.pyplot as plt 

from utils import panel_matrices
from get_data import quota
from sdid import SDID
from vcov import Variance
from plots import Plots 
from summary import Summary


class Synthdid(SDID, Variance, Plots, Summary):
	def __init__(self, data, unit="unit", time="time", treatment="treatment", outcome="outcome",
			  covariates=None):
		self.data=data
		self.unit, self.time=unit, time
		self.treatment, self.outcome=treatment, outcome
		self.covariates=covariates
		self.se=None
		self.data_ref, self.ttime = panel_matrices(
			data, unit, time, treatment, outcome, covariates=covariates)
