from get_data import birthrates, num_births

from synthdid import Synthdid

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 


# (1) Birthrate fit ------------------------------------------------------------------------------ #
df = birthrates()

# SDID ---------------------------------------------------- #
# SDID_result = Synthdid(df, 
# 					   "State", "Year", "treated", "Birthrate").fit().vcov(method="placebo", n_reps=200).summary().summary2
# print(SDID_result)

# SDID_outcome_plot = Synthdid(df,
# 							 "State", "Year", "treated", "Birthrate").fit().plot_outcomes()
# plt.show()
# SDID_weights_plot = Synthdid(df,
# 							 "State", "Year", "treated", "Birthrate").fit().plot_weights()
# plt.show()

# SC ------------------------------------------------------ #
# SC_result = Synthdid(df, 
# 					   "State", "Year", "treated", "Birthrate").fit(
# 						   synth=True
# 					   ).vcov(method="placebo", n_reps=200).summary().summary2
# print(SC_result)

# SDID_outcome_plot = Synthdid(df,
# 							 "State", "Year", "treated", "Birthrate").fit(
# 								 synth=True, omega_intercept=False
# 							 ).plot_outcomes()
# plt.show()

# SDID_weights_plot = Synthdid(df,
# 							 "State", "Year", "treated", "Birthrate").fit(
# 								 synth=True, omega_intercept=False
# 							 ).plot_weights()
# plt.show()

# DID ----------------------------------------------------- #
# DID_result = Synthdid(df, 
# 					  "State", "Year", "treated", "Birthrate").fit(did=True).vcov(method="placebo", n_reps=200).summary().summary2
# print(DID_result)

# DID_outcome_plot = Synthdid(df,
# 							"State", "Year", "treated", "Birthrate").fit(did=True).plot_outcomes()
# plt.show()
# DID_weights_plot = Synthdid(df,
# 							"State", "Year", "treated", "Birthrate").fit(did=True).plot_weights()
# plt.show()


# With Covariates --------------------------------------------------------------------------- #
pop = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["population_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

female = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["female_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

marriage = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["marriage_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

migration = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["migration_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

tax = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["tax_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

finance = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["finance_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

pop_female = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["population_zscore", "female_zscore"]).fit().vcov(
				   method="placebo", n_reps=1000).summary().summary2

print(pop)
print()
print(female)
print()
print(marriage)
print()
print(migration)
print()
print(tax)
print()
print(finance)
print()
print(pop_female)

# projected --------------------------------------------------------------- # 
pop = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["population_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

female = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["female_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

marriage = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["marriage_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

migration = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["migration_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

tax = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["tax_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

finance = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["finance_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

pop_female = Synthdid(df, "State", "Year", "treated", "Birthrate", 
			   covariates=["population_zscore", "female_zscore"]).fit(
				   cov_method="projected"
			   ).vcov(
				   method="placebo", n_reps=1000).summary().summary2

print("-"*99)
print(pop)
print()
print(female)
print()
print(marriage)
print()
print(migration)
print()
print(tax)
print()
print(finance)
print()
print(pop_female)