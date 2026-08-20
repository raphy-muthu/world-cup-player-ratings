import pandas as pd

final = pd.read_csv("data/processed/wc_fbref_final.csv")
print(final.shape)
print(final.columns.tolist())

for col in ['tournament_goals', 'assists', 'shots', 'shots_on_target', 'yellow_cards', 'red_cards']:
    print(col, (final[col] > 0).mean(), final[col].describe())

