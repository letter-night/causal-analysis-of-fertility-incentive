import pandas as pd
import numpy as np

from optimizer import Optimize
from plot import Plot
from variance import Variance
from summary import Summary


class SynthDID(Optimize, Plot, Variance, Summary):
	"""
	Synthetic Difference in Differences
	df				: pandas.DataFrame
	pre_term		: term before treatment
	post_term		: term after treatment
	treatment_unit:		treatment columns names list
	
	[example]
	df = fetch_birthrates() (or fetch_num_births())
	sdid = SynthDID(df, [2000, 2023], [2024, 2024], ["Incheon"])
	sdid.fit()
	"""

	def __init__(
			self, df, pre_term, post_term, treatment_unit:list, random_seed=0, **kwargs
	):
		# first val
		self.df=df
		self.pre_term=pre_term 
		self.post_term=post_term
		self.random_seed=random_seed
		# able to change
		self.treatment=treatment_unit
		self.control = [col for col in df.columns if col not in self.treatment]
		self._divide_data()

		# params
		self.hat_zeta = None
		self.base_zeta = None
		self.hat_omega_ADH = None
		self.hat_omega = None
		self.hat_lambda = None
		self.sdid_se = None
		self.sc_se = None
		self.did_se = None

	def _divide_data(self):

		self.Y_pre_c = self.df.loc[self.pre_term[0] : self.pre_term[1], self.control]
		self.Y_pre_t = self.df.loc[self.pre_term[0] : self.pre_term[1], self.treatment]

		self.Y_post_c = self.df.loc[self.post_term[0] : self.post_term[1], self.control]
		self.Y_post_t = self.df.loc[self.post_term[0] : self.post_term[1], self.treatment]

		self.n_treat = len(self.treatment)
		self.n_post_term = len(self.Y_post_t)
	
	def fit(
			self,
			model="all",
			zeta_type="base",
			additional_X=pd.DataFrame(),
			additional_y=pd.DataFrame(),
	):
		
		self.base_zeta = self.est_zeta(self.Y_pre_c)

		if zeta_type == "base":
			self.zeta = self.base_zeta
		
		else:
			print(f"your choice : {zeta_type} is not supported.")
			self.zeta = self.base_zeta


		self.hat_omega = self.est_omega(self.Y_pre_c, self.Y_pre_t, self.zeta)
		self.hat_omega_ADH = self.est_omega_ADH(
			self.Y_pre_c,
			self.Y_pre_t,
			additional_X=additional_X,
			additional_y=additional_y
		)
		self.hat_lambda = self.est_lambda(self.Y_pre_c, self.Y_post_c)

	
	def did_potential_outcome(self):
		"""
		return potential outcome
		"""
		Y_pre_c = self.Y_pre_c.copy()
		Y_pre_t = self.Y_pre_t.copy()
		Y_post_c = self.Y_post_c.copy()
		Y_post_t = self.Y_post_t.copy()

		if type(Y_pre_t) != pd.DataFrame:
			Y_pre_t = pd.DataFrame(Y_pre_t)
		
		if type(Y_post_t) != pd.DataFrame:
			Y_post_t = pd.DataFrame(Y_post_t)
		
		Y_pre_t["did"] = Y_pre_t.mean(axis=1)

		base_trend = Y_post_c.mean(axis=1)

		Y_post_t["did"] = base_trend + (
			Y_pre_t.mean(axis=1).mean() - Y_pre_c.mean(axis=1).mean()
		)

		return pd.concat([Y_pre_t["did"], Y_post_t["did"]], axis=0)
	

	def sc_potential_outcome(self):
		return pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_ADH)
	

	def sdid_trajectory(self):
		hat_omega = self.hat_omega[:-1]
		Y_c = pd.concat([self.Y_pre_c, self.Y_post_c])
		n_features = self.Y_pre_c.shape[1]
		start_w = np.repeat(1 / n_features, n_features)

		_intercept = (start_w - hat_omega) @ self.Y_pre_c.T @ self.hat_lambda

		return Y_c.dot(hat_omega) + _intercept
	
	def sdid_potential_outcome(self):
		Y_pre_c = self.Y_pre_c.copy()
		Y_post_c = self.Y_post_c.copy()
		hat_omega = self.hat_omega[:-1]

		base_sc = Y_post_c @ hat_omega
		pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
		pre_control_base = Y_pre_c @ hat_omega @ self.hat_lambda

		pre_outcome = Y_pre_c.dot(hat_omega)

		post_outcome = base_sc + pre_treat_base - pre_control_base

		return pd.concat([pre_outcome, post_outcome], axis=0)
	

	def target_y(self):
		return self.df.loc[self.pre_term[0] : self.post_term[1], self.treatment].mean(
			axis=1
		)
	
	def estimated_params(self, model="sdid"):
		Y_pre_c_intercept = self.Y_pre_c.copy()
		Y_post_c_intercept = self.Y_post_c.copy()
		Y_pre_c_intercept["intercept"] = 1
		Y_post_c_intercept["intercept"] = 1
		
		if model == "sdid":
			return (
				pd.DataFrame(
					{
						"features": Y_pre_c_intercept.columns,
						"sdid_weights": np.round(self.hat_omega, 3),
					}
				),
				pd.DataFrame(
					{
						"time": Y_pre_c_intercept.index,
						"sdid_weight": np.round(self.hat_lambda, 3)
					}
				)
			) 
		
		elif model == "sc":
			return pd.DataFrame(
				{
					"features": self.Y_pre_c.columns,
					"sc_weight": np.round(self.hat_omega_ADH, 3)
				}
			)
	
	def hat_tau(self, model="sdid"):
		"""
		return ATT
		"""
		result = pd.DataFrame({"actual_y": self.target_y()})
		post_actual_treat = result.loc[self.post_term[0] :, "actual_y"].mean()

		if model == "sdid":
			result["sdid"] = self.sdid_trajectory()

			pre_sdid = result["sdid"].head(len(self.hat_lambda)) @ self.hat_lambda
			post_sdid = result.loc[self.post_term[0] :, "sdid"].mean()

			pre_treat = (self.Y_pre_t.T @ self.hat_lambda).values[0]
			counterfactual_post_treat = pre_treat + (post_sdid - pre_sdid)
		
		elif model == "sc":
			result["sc"] = self.sc_potential_outcome()
			post_sc = result.loc[self.post_term[0] :, "sc"].mean()
			counterfactual_post_treat = post_sc 
		
		elif model == "did":
			Y_pre_t = self.Y_pre_t.copy()
			Y_post_t = self.Y_post_t.copy()

			if type(Y_pre_t) != pd.DataFrame:
				Y_pre_t = pd.DataFrame(Y_pre_t)
			
			if type(Y_post_t) != pd.DataFrame:
				Y_post_t = pd.DataFrame(Y_post_t)
			
			# actual treat
			post_actual_treat = (
				Y_post_t.mean(axis=1).mean() - Y_pre_t.mean(axis=1).mean()
			)
			counterfactual_post_treat = (
				self.Y_post_c.mean(axis=1).mean() - self.Y_pre_c.mean(axis=1).mean()
			)
		
		return post_actual_treat - counterfactual_post_treat
	
	def cal_se(self, algo="placebo", replications=200):

		sdid_var, sc_var, did_var = self.estimate_variance(
			algo=algo, replications=replications
		)

		self.sdid_se = np.sqrt(sdid_var)
		self.sc_se = np.sqrt(sc_var)
		self.did_se = np.sqrt(did_var)


if __name__ == "__main__":
	from sample_data import fetch_birthrates, fetch_num_births

	df = fetch_birthrates()

	PRE_TEREM = [2000, 2023]
	POST_TEREM = [2024, 2024]
	TREATMENT = ["Incheon"]

	sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
	sdid.fit()
	# sdid.plot(model="sdid")
	# sdid.plot(model="sc")
	sdid.plot(model="did")

