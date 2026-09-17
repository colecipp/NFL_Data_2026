import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import nfl_data_py as nfl


class NFLData:
    NFL = pd.read_excel("path of NFL data set")
    # def __init__(self, year):
    #     self.year = year
    #     self.data = nfl.load_pbp_data(year)
    def print_columns(self):
        print(NFL.columns.tolist())

    def calc_pythagExpect(self):
        NFL25 = NFL[["visitingteam, hometeam, visitorpoints, homepoints, date"]]
        #gather when the team is both the home team
        #and away team in order to use the full set
        #of data to calculate the Pythagorean expectation
        NFL25 = NFL25.rename(columns={'visitorpoints':'visR', 'TDsScored':'TD'})
        #unexplained renaming of columns
        print(NFL25)
        NFL25['homeWins'] = np.where(NFL25['homepoints'] > NFL25['visR'], 1, 0)
        NFL25['awayWins'] = np.where(NFL25['visR'] > NFL25['homepoints'], 1, 0)
        #calculating the wins and losses, quite clear
        #will need to tweak for ties
        NFL25['count']=1
        print(NFL25)
        NFLHome = NFL25.groupby('homeTeam')['homeWins','homePoints','visitingPoints', 'count'].sum().reset_index()
        #groups them up and counts them
        #We group by home team
        #to obtain the sum of wins and runs (scored and conceded) 
        #and also the counter variable to show how many games were played 
        NFLHome = NFLHome.rename(columns={'HomeTeam':'team','VisR':'VisRh','HomR':'HomRh','count':'Gh'})
        #renaming for some reason
        NFLHome = NFL25.groupby('visitingTeam')['awayWins','homePoints','visitingPoints', 'count'].sum().reset_index()
        #now away wins
        NFLHome = NFLHome.rename(columns={'awayTeam':'team','VisR':'VisRh','HomR':'HomRh','count':'Gh'})
        return NFL25
