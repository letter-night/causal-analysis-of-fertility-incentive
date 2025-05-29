import pandas as pd

def birthrates() -> pd.DataFrame:
	return pd.read_csv("./data/data_birthrate_cov.csv")

def num_births() -> pd.DataFrame:
	return pd.read_csv("./data/data_num_births_cov.csv")

