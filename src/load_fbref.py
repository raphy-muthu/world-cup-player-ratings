"""
DOCSTRING
"""

import pandas as pd

fbref = pd.read_csv("data/raw/players_data_light-2025_2026.csv")

print(fbref.head())


def nationClean(nationValue):
    if pd.isna(nationValue):
        return None

    clean_nation = nationValue.split()
    if len(clean_nation) != 2:
        return None
    else:
        return(clean_nation[1])
    


fbref["nationality_code"] = fbref["Nation"].apply(nationClean)

print(fbref["nationality_code"].isna().sum())

# 3 players have missing nations (Nathan Mbala, Luis Orejuela, Yael Trepy). Can get nation from google
print(fbref[fbref['nationality_code'].isna()][['Player', 'Nation', 'Squad', "Pos"]])