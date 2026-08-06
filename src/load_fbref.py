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

fbref.to_csv("data/processed/fbref_cleaned.csv", index=False)
print("\nSaved to data/processed/fbref_cleaned.csv")

fbrefGrouped = fbref.groupby("Player").size()

duplicateNames = fbrefGrouped[fbrefGrouped > 1]
print(f"Duplicate Names: {duplicateNames}")

fbrefFilter = fbref[fbref["Player"].isin(duplicateNames.index)]

# WHEN DOING FUZZY MATCHING: Must also match players from duplicateNames with the club to avoid statistical overlap
print(fbrefFilter.sort_values("Player"))

fbref = fbref.drop(columns=['PK_stats_shooting', 'PKatt_stats_shooting', 'CrdY_stats_misc', 'CrdR_stats_misc'])
count_columns = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt',
    'CrdY', 'CrdR', 'G+A-PK', 'Sh', 'SoT', 'Crs', 'TklW', 'Int', 'Fld',
    'Fls', 'OG', 'GA', 'SoTA', 'Saves', 'W', 'D', 'L', 'CS',
    'PKatt_stats_keeper', 'PKA', 'PKsv', 'PKm'
]

# Handle mid-season transfers: combine split rows into one per player
# 151 players have two rows in FBref because of a mid-season club transfer

# Columns that represent counts and should be SUMMED across a player's rows
count_columns = [
    'MP', 'Starts', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR',
    'TklW', 'Int', 'Crs', 'Fls'
]

agg_rules = {col: 'sum' for col in count_columns}
agg_rules['Squad'] = lambda squads: ' / '.join(squads.unique())
agg_rules['nationality_code'] = 'first'
agg_rules['Nation'] = 'first'
# add 'first' for any other non-numeric column you want to keep (e.g., Pos, Age)

fbref_aggregated = fbref.groupby('Player', as_index=False).agg(agg_rules)

print(f"Rows before aggregation: {len(fbref)}")
print(f"Rows after aggregation: {len(fbref_aggregated)}")
print(f"Unique player names: {fbref['Player'].nunique()}")
assert len(fbref_aggregated) == fbref['Player'].nunique(), "Aggregation didn't collapse duplicates correctly!"

# Spot check a known transferred player to confirm it worked as expected
print(fbref_aggregated[fbref_aggregated['Player'] == 'Adama Traoré'])

fbref_aggregated.to_csv("data/processed/fbref_cleaned.csv", index=False)
print("Saved to data/processed/fbref_cleaned.csv")