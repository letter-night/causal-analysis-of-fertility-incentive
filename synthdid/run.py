import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath("../"))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', 
# '_mpl-gallery', '_mpl-gallery-nogrid', 
# 'bmh', 'classic', 'dark_background', 
# 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
# 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 
# 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 
# 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 
# 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 
# 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']

plt.style.use('ggplot')

from model import SynthDID
from optimizer import Optimize
from plot import Plot
from variance import Variance
from summary import Summary
from sample_data import fetch_birthrates, fetch_num_births

# 1. birthrates ATT estimation ############################################

df = fetch_birthrates()

# print(df.head())

PRE_TEREM = [2000, 2023]
POST_TEREM = [2024, 2024]
TREATMENT = ["Incheon"]

## (1) Fitting the model (SDID, SC, DID) ------------------------------------------------------ #
sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
sdid.fit()

## (2) Visualize the estimation of SDID, SC, DID ---------------------------------------------- #
# sdid.plot(model="sdid")
# sdid.plot(model="sc")
# sdid.plot(model="did")

## (3) Confidence interval based on the placebo method -------------------------------------- #
# sdid.cal_se(algo="placebo", replications=200)

# print(sdid.summary(model="sdid"))
# print(sdid.summary(model="sc"))
# print(sdid.summary(model="did"))

## (4) Visualize the estimated parameters ---------------------------------------------------- #
### 1) omega - Unit weights ----------------------------------------------------------------- #
hat_omega = sdid.estimated_params(model="sc")

hat_omega_sdid, hat_lambda_sdid = sdid.estimated_params()

omega_result = pd.merge(
	hat_omega, hat_omega_sdid, left_on="features", right_on="features", how="left"
)

# print(omega_result)

# fig = plt.figure()
# fig.set_figwidth(11)
# fig.set_figheight(4)
# ax = fig.add_subplot(1, 1, 1)
# width = 0.29
# ind = np.arange(len(omega_result))

# ax.bar(ind - width, omega_result["sc_weight"], width, label="sc")
# ax.bar(ind, omega_result["sdid_weights"], width, label="sdid")

# ax.set_xticks(ind)
# ax.set_xticklabels(omega_result["features"].values)
# ax.legend()
# ax.set_ylabel("omega weight")
# ax.set_title("Estimated unit weights: SDID vs SC")

# fig.tight_layout()
# plt.xticks(rotation=11)
# plt.show()

### 2) lambda - Time weights --------------------------------------------------------------- #
# print(hat_lambda_sdid)

# fig = plt.figure()
# fig.set_figwidth(11)
# fig.set_figheight(4)
# ax = fig.add_subplot(1, 1, 1)
# width = 0.29
# ind = np.arange(len(hat_lambda_sdid))

# ax.bar(ind, hat_lambda_sdid["sdid_weight"], width, label="sdid time weight")

# ax.set_xticks(ind)
# ax.set_xticklabels(hat_lambda_sdid["time"].values)
# ax.legend()
# ax.set_ylabel("lambda weight")
# ax.set_title("Estimated time weights: SDID")

# fig.tight_layout()
# plt.show()


# 2. number of births ATT estimation #######################################

df = fetch_num_births()

print(df.head())

PRE_TEREM = [2000, 2023]
POST_TEREM = [2024, 2024]
TREATMENT = ["Incheon"]

## (1) Fitting the model (SDID, SC, DID) ------------------------------------------------------ #
sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
sdid.fit()

## (2) Visualize the estimation of SDID, SC, DID ---------------------------------------------- #
# sdid.plot(model="sdid")
# sdid.plot(model="sc")
# sdid.plot(model="did")

## (3) Confidence interval based on the placebo method -------------------------------------- #
# sdid.cal_se(algo="placebo", replications=200)

# print(sdid.summary(model="sdid"))
# print(sdid.summary(model="sc"))
# print(sdid.summary(model="did"))

## (4) Visualize the estimated parameters ---------------------------------------------------- #
### 1) omega - Unit weights ----------------------------------------------------------------- #
hat_omega = sdid.estimated_params(model="sc")

hat_omega_sdid, hat_lambda_sdid = sdid.estimated_params()

omega_result = pd.merge(
	hat_omega, hat_omega_sdid, left_on="features", right_on="features", how="left"
)

# print(omega_result)

fig = plt.figure()
fig.set_figwidth(11)
fig.set_figheight(4)
ax = fig.add_subplot(1, 1, 1)
width = 0.29
ind = np.arange(len(omega_result))

ax.bar(ind - width, omega_result["sc_weight"], width, label="sc")
ax.bar(ind, omega_result["sdid_weights"], width, label="sdid")

ax.set_xticks(ind)
ax.set_xticklabels(omega_result["features"].values)
ax.legend()
ax.set_ylabel("omega weight")
ax.set_title("Estimated unit weights: SDID vs SC")

fig.tight_layout()
plt.xticks(rotation=11)
plt.show()

### 2) lambda - Time weights --------------------------------------------------------------- #
# print(hat_lambda_sdid)

fig = plt.figure()
fig.set_figwidth(11)
fig.set_figheight(4)
ax = fig.add_subplot(1, 1, 1)
width = 0.29
ind = np.arange(len(hat_lambda_sdid))

ax.bar(ind, hat_lambda_sdid["sdid_weight"], width, label="sdid time weight")

ax.set_xticks(ind)
ax.set_xticklabels(hat_lambda_sdid["time"].values)
ax.legend()
ax.set_ylabel("lambda weight")
ax.set_title("Estimated time weights: SDID")

fig.tight_layout()
plt.show()