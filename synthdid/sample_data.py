import pandas as pd

def fetch_birthrates() -> pd.DataFrame:

	_raw = pd.read_csv("./data/birthrates.csv")

	_raw.index = [i for i in range(2000, 2025)]

	return _raw.loc[2000: ]


def fetch_num_births() -> pd.DataFrame:

	_raw = pd.read_csv("./data/num_births.csv")

	_raw.index = [i for i in range(2000, 2025)]

	return _raw.loc[2000: ]
