import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


contracts = pd.read_excel('NFL_Contract_Data_v2.xlsx', sheetname='Contracts').set_index('ID')
pff = pd.read_excel('PFF 0-100 Grades NFLSeason2006to2016.xlsx')
players = pd.read_excel('NFL Contract Data_v1.xlsx', sheetname='Players')
contracts_players = pd.merge(contracts, players, 'inner', left_on='player_id', right_on='id')
contracts_players['Name'] = contracts_players['Name_y'].apply(lambda x: x.replace('\\', '') if type(x)==str else x)


last_season = lambda x: pff[(pff.player==x.Name) & (pff.season < x.year_signed)].season.max()
contracts_players['last_season'] = contracts_players.apply(last_season, axis=1)


contracts_players_perf = pd.merge(contracts_players, pff, 'left',
                                  left_on=['Name', 'last_season'],
                                  right_on=['player', 'season'],
                                  suffixes=['_otc', '_pff'])


contracts_players_perf = contracts_players_perf.drop(['Name_x', 'Name_y', 'id', 'last_season'], axis=1)
contracts_players_perf.to_csv('contracts_players_perf.csv')

