# Import Libraries

import pandas as pd                                                      # Data manipulation
import numpy as np                                                      # Numerical operations and arrays
import datetime                                                         # Date and time handling
import matplotlib.pyplot as plt                                         # Plotting and visualization
import matplotlib.cm as cm                                              # Colormap handling for visualizations
import seaborn as sns                                                   # Statistical data visualization
import nfl_data_py as nfl                                               # NFL data retrieval and processing

# Load NFL Data
season_years = [2022, 2023]  # Define the seasons to analyze

# Import Team Data
df_teams = nfl.import_team_desc()
display(df_teams)
# Import Game Data
df_games = nfl.import_schedules(season_years)

# Import Player Statistics
df_players = nfl.import_seasonal_data(season_years)

# # Merge Data
# nfl_data = df_games.merge(df_teams, left_on='home_team', right_on='team_abbr', how='left')
# nfl_data = nfl_data.merge(df_teams, left_on='away_team', right_on='team_abbr', suffixes=('_home', '_away'))

# # Convert Date Column to Datetime Format
# nfl_data['gameday'] = pd.to_datetime(nfl_data['gameday'])

# # Sort by Date
# nfl_data = nfl_data.sort_values(by='gameday')

# # Compute Total Points Scored
# nfl_data['total_points'] = nfl_data['home_score'] + nfl_data['away_score']

# # Compute Win Percentage for Teams
# win_data = nfl_data[['home_team', 'home_score', 'away_score']]
# win_data['win'] = np.where(win_data['home_score'] > win_data['away_score'], 1, 0)
# team_wins = win_data.groupby('home_team')['win'].mean().reset_index()
# team_wins.rename(columns={'win': 'Win_Percentage'}, inplace=True)

# # Compute Total Points Per Team
# total_points = nfl_data.groupby('home_team')['total_points'].sum().reset_index()
# total_points.rename(columns={'total_points': 'Total_Points'}, inplace=True)

# # Merge Win Percentage and Total Points
# total_stats = total_points.merge(team_wins, on='home_team')

# # Assign Colors to Teams for Visualization
# teams = total_stats['home_team'].unique()
# colors = cm.get_cmap('tab10', len(teams))
# team_color_map = {team: colors(i) for i, team in enumerate(teams)}

# # Plot Win Percentage vs. Total Points Scored
# plt.figure(figsize=(10,6))
# for team in teams:
#     subset = total_stats[total_stats['home_team'] == team]
#     plt.scatter(subset['Total_Points'], subset['Win_Percentage'],
#                 color=team_color_map[team], label=team, edgecolors='black')

# plt.xlabel('Total Points Scored')
# plt.ylabel('Win Percentage')
# plt.title('Win Percentage vs Total Points Scored')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.show()
