# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:15:07 2020
@author: Charles Joseph
Analyzing the effect that space on EFG% in the 2014-2015 season
Data is available on kaggle. There was a little bit of cleaning done. 
"""
from shotLog_functions import *
from sequential_forward_selection import *

ploton = False
    
#load datasets
shot_logs = pd.read_csv("C:/Users/charl/Desktop/Udacity_DataScience/Data/basketball/shot_logs.csv")
all_seasons = pd.read_csv("C:/Users/charl/Desktop/Udacity_DataScience/Data/basketball/all_seasons.csv")


# GROUP DATA BY player_name. Use the max height bc players repeat. 
heights_df = all_seasons.groupby(['player_name']).player_height.agg(player_height = 'max') 
# make the names all lower case so that they can be matched more easily
heights_df.index = [x.casefold() for x in heights_df.index.values]
heights_df = heights_df.reset_index().rename(columns = {'index':'player_name'})
# MERGE PLAYER'S HEIGHT ONTO SHOT LOG DATA
shots_and_heights_df = pd.merge(left = shot_logs, right = heights_df, on='player_name',how='left')

## create a copy of the data and delete columns we don't plan on using. 
X = shots_and_heights_df.copy()
X = X.drop(columns=['MATCHUP','LOCATION','W','FINAL_MARGIN','PERIOD','GAME_CLOCK'])

#GET PLAYERS WHO HAD MORE THAN 300 MADE BASKETS
shots_made = X.loc[X['SHOT_RESULT']=='made'] #get only rows with a made shot
shots_made = shots_made.groupby('player_name')['SHOT_RESULT'].agg(SHOTS_MADE='count') #get total makes for each player
shots_made = shots_made.loc[shots_made.SHOTS_MADE >= 300] #keep rows where there were over 300 makes

#names of valid players as list
valid_players = shots_made.index 


## CALCULATE EFG, Avg Shot Distance and Avg Closest Defender Distance (Avg Space)
# get season FG%
FG = X.groupby('player_name').FGM.agg(FG='mean')
# made  3P%
EFG = X.groupby('player_name')['PTS'].agg(EFG = lambda x: np.mean(x==3))
# EFG% = FG% + 0.5 *3P%
EFG.EFG = 0.5*EFG.EFG+FG.FG

# Get average defended distance and join to EFG df
Avg_Dist = X.groupby('player_name').CLOSE_DEF_DIST.agg(Avg_Dist='mean')
EFG = EFG.join(Avg_Dist)

# get shot distance and join to EFG df
Shot_Dist = X.groupby('player_name').SHOT_DIST.agg(Avg_Shot = 'mean')
EFG = EFG.join(Shot_Dist)

# get player heights and join to EFG df
heights = X.groupby('player_name').player_height.agg(player_height = 'max')
EFG = EFG.join(heights)

# filter using valid players list
EFG = EFG.loc[valid_players]
standardized_EFG = (EFG-EFG.mean())/EFG.std()
standardized_EFG = standardized_EFG.sort_values(by='Avg_Dist',ascending=False)
print(standardized_EFG.head())

## Ended up settling on a 2nd degree polynomial because the adj R2 was better
## than the adj R2 of the 3rd degree. And for the 3rd degree model, only the 
## second order term has a confidence interval excluding 0. 
## Tried using height as a predictor but it wasn't very helpful
predictors = standardized_EFG.columns.tolist().copy()
predictors.remove('EFG')
response   = 'EFG'

model_ = sequential_forward_selection(data = standardized_EFG, predictors = predictors, response = response, max_length = 10, numerical_predictors = predictors, entry_limit = 0.10)


lm2 = polyFit(standardized_EFG.Avg_Dist.values.reshape(-1,1), \
              standardized_EFG.EFG.values.reshape(-1,1), 2)
print(lm2.summary())
x0, x1, x2 = lm2.params

# create plots
lm_diagnostics(lm2)
lm_plot_rankx(lm2, standardized_EFG, 'Avg_Dist', 'EFG', False,\
              'Offensive Players with Most Separation',\
              'Studentized Closest Defender Dist', 'Studentized EFG%','best_space_offense.jpg')
lm_plot_rankx(lm2, standardized_EFG, 'Avg_Dist', 'EFG', True,\
              'Offensive Players with Worst Separation',\
              'Studentized Closest Defender Dist', 'Studentized EFG%','worst_space_offense.jpg')
lm_plot_ranky(lm2, standardized_EFG, 'Avg_Dist', 'EFG', False,\
              'Players with Best EFG%','Studentized Closest Defender Dist',\
              'Studentized EFG%','best_EFG_offense.jpg')
lm_plot_ranky(lm2, standardized_EFG, 'Avg_Dist', 'EFG', True,\
              'Players with Worst EFG%','Studentized Closest Defender Dist',\
              'Studentized EFG%','worst_EFG_offense.jpg')


###########################################################
########## DEFENSE     ####################################
###########################################################

## Get the players responsible for the most 'constest' (Roughly top 100)
Defense = X.groupby('CLOSEST_DEFENDER').CLOSEST_DEFENDER_PLAYER_ID.agg(CONTESTS = 'count').sort_values(by='CONTESTS',ascending=False)

Defense.index = fix_names([x.casefold() for x in Defense.index])
valid_players = Defense[Defense.CONTESTS>=450].index


# Determine the Effective Field Goal percentage of players being defended by pooling the makes 
# and misses of the players who were defended by a particular defender. For example
# player A is closest defender to player B for 5 shots: 2 pts: 1/5, 3 pts: 0/0 
# player A is also the closest defender to player C for 5 shost: 2 pts 0/3, 3pts: 2/2
# player A forced a Effective Field Goal percentage of 40%

# Calculate lumped EFG percentage of players attempting field goals closest to person here
FG = X.groupby('CLOSEST_DEFENDER')['FGM'].agg(FG='mean')
EFG = X.groupby('CLOSEST_DEFENDER')['PTS'].agg(EFG = lambda x: np.mean(x==3))
EFG.EFG = 0.5*EFG.EFG+FG.FG

# Calculate average distance away from person attempting the shot
Avg_Dist = X.groupby('CLOSEST_DEFENDER').CLOSE_DEF_DIST.agg(Avg_Dist='mean')
EFG = EFG.join(Avg_Dist)

# Average of how far away from basket shots were taken
Shot_Dist = X.groupby('CLOSEST_DEFENDER').SHOT_DIST.agg(Avg_Shot = 'mean')
EFG = EFG.join(Shot_Dist)

# only get data from players who showed up as closest defender often enough
# to bring down the sample to about 100 players
# Put defender names in firstname lastname format 
EFG = EFG.reset_index().rename(columns={'CLOSEST_DEFENDER':'player_name'})
EFG.player_name = [x.casefold() for x in EFG.player_name.values]
EFG.player_name = fix_names(EFG.player_name.values)

EFG = pd.merge(left = EFG, right = heights_df, on='player_name',how='left')
EFG = EFG.set_index('player_name')
EFG = EFG.loc[valid_players]
# standardize df by centering and scaling by std
standardized_EFG = (EFG-EFG.mean())/EFG.std()

## Build model
defense_model, formula = forward_selected(standardized_EFG,'EFG')
print(defense_model.summary())


lm_diagnostics(defense_model,"Studentized Closest Defender Dist.")


lm_plot_rankx(defense_model, standardized_EFG, 'Avg_Dist', 'EFG', False,\
              'Furthest Defenders', 'Studentized Closest Defender Dist.',\
              'Studentized EFG%','worst_space_defense.png')

lm_plot_rankx(defense_model, standardized_EFG, 'Avg_Dist', 'EFG', True,\
              'Closest Defenders', 'Studentized Closest Defender Dist.','Studentized EFG%',\
              'best_space_defense.png')

    
lm_plot_ranky(defense_model, standardized_EFG, 'Avg_Dist', 'EFG',True,\
              'Defenders Associated with Worst EFGs', 'Studentized Offensive Player Dist.', 'Studentized EFG%',\
              'best_defense_efg.png')

lm_plot_ranky(defense_model, standardized_EFG, 'Avg_Dist', 'EFG',False,\
              'Defenders Associated with Best EFGs', 'Studentized Offensive Player Dist.', 'Studentized EFG%',\
              'worst_defense_efg.png')    
    
