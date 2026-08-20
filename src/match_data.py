import pandas as pd

wc = pd.read_csv("data/processed/wc_merged.csv")
fbref = pd.read_csv("data/processed/fbref_cleaned.csv")

print(wc.shape)
print(fbref.shape)

print(wc.columns.tolist())
print(fbref.columns.tolist())

finalData = wc.merge(fbref, on="name_norm", how="left")

print(finalData["Player"].notna().sum())
print(finalData["Player"].isna().sum())

final = pd.read_csv('data/processed/wc_fbref_final.csv')

print(final['position'].value_counts())
print()
print(final.groupby('position')['minutes_played'].describe())