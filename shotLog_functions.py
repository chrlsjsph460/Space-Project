# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:08:23 2020

@author: charl
"""
#import packages for processing data: numpy, pandas, matplotlib.pyplot, 
# seasborn for plotting, PolynomailFeatures for polynomial linear models
# statsmodel and statsmodels.formula for R style linear models, and qqplot


import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d # might use this to visualize 3 dimensions
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # might use this for linear fit
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.gofplots import qqplot
plt.style.use('fivethirtyeight')

def forward_selected(data, response):
    """Linear model designed by forward selection.
    # from: https://planspace.org/20150423-forward_selection_with_statsmodels/ 
    
    Inputs
    data : pandas DataFrame with all possible predictors and response
    response: string, name of response column in data

    Outputs
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    
    # create set containing all column names
    remaining = set(data.columns) 
    remaining.remove(response)
    
    # create empty list to store variables that we select
    selected = []
    # initialize current and best_new_score to zero
    current_score, best_new_score = 0.0, 0.0
    # while the set of column names left out of the model is not empty and 
    # the current score is the best get the score of the model after adding each 
    # column in remaining and append the results to a list. Sort the list and add the 
    # column with the best score to the model if model improves. 
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model, formula

# Function for polynomial fit with 
def polyFit(X, y, deg=2):
    poly = PolynomialFeatures(degree= deg)
    X_poly = poly.fit_transform(X)
    
    lm = sm.OLS(y, X_poly).fit()
    
    return lm

def studentize_df(df):
    return (df - df.mean())/df.std()

# Function to reorder names
def fix_names(names_list):
    """ 
    Takes list of names in format lastname, firstname and returns list of names in 
    format firstname lastname.
    """
    
    #using list comprehension. Take name. 1. get rid of comma and space using split. 
    #This creates 2 entries in each list. [::-1] puts them in reverse order
    #join them with a space between. 
    
    fixed_names = [" ".join(name.split(", ")[::-1]) for name in names_list]
    return fixed_names

def lm_diagnostics(model, xlabel="x"):
    ''' create plot of standardized residuals
        create plot of leverage
        create plot of data and fit
    '''
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = False, sharey = False)
    axs[0].scatter(np.arange(len(model.resid)), studentize_df(model.resid),color='r')
    axs[0].set_title("Studentized Residuals vs {}".format(xlabel))
    axs[0].set_ylabel("Studentized Residuals")
    axs[0].set_xlabel(xlabel)
    #axs[0].show()
    qqplot(model.resid,line='45', ax = axs[1])
    axs[1].set_title("Residual QQ-Plot" )
    axs[1].set_xlabel("Theoretical Quantiles")
    axs[1].set_ylabel("Sample Quantiles")

def lm_plot_rankx(model, df, xname, yname, ascending, title, xlabel, ylabel,fname):
    '''
    Create plots used to show top five offense and defense players
    Inputs:
        model: statsmodel linear model
        df   : dataframe holding data used in model
        xname: name of predictor column
        yname: name of response column
        
    
    '''
    #sor df by x
    sorted_df = df.sort_values(by=xname, ascending=ascending)
    #get the first 5 names and corresponding (x,y) values
    top5_names = sorted_df.index[:5]
    top5_y = sorted_df[yname].values[:5]
    top5_x = sorted_df[xname].values[:5]
    #assign unique markers for the 5 names
    marker = ['d','x','^','<','>']

    fig, ax = plt.subplots(1,1)    
    #Create a scatter plot using the remaining (x,y) pairs
    ax.scatter(sorted_df[xname].iloc[5:], sorted_df[yname].iloc[5:],color = 'k')
    #create x- and y-labels
    ax.set_ylabel(yname)
    ax.set_xlabel(xname)
    #use for loop to plot top 5 
    for name, y, x, marker in zip(top5_names, top5_y,top5_x, marker):
        ax.plot(x, y, marker, label =name)
    #limit your axis limits to 3 standard deviations from the mean and label them    
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    color_top = ['b', 'r', 'g','c','m']

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    #label for model predictions. Model is linear for defense and quadratic for offense.
    if len(model.params) == 2:
        label_string = f'EFG = {round(model.params[1],3)}*{xname}+{round(model.params[0],3)}'
    else: 
        label_string = f'EFG = {round(model.params[2],3)}*{xname}^2 + {round(model.params[1],3)}*{xname}+{round(model.params[0],3)}'

    
    ax.plot(df[xname], model.fittedvalues, label=label_string)
    ax.legend(framealpha = 0,fontsize = 'x-small', loc = 'lower left')
    ax.set_title(title)   
    fig.savefig(fname,bbox_inches='tight', pad_inches=0.5)
    
def lm_plot_ranky(model, df, xname, yname, ascending, title, xlabel, ylabel,fname):
    '''
    Create plots used to show top five offense and defense players
    Inputs:
        model: statsmodel linear model
        df   : dataframe holding data used in model
        xname: name of predictor column
        yname: name of response column
        
    Print only First Initial and Last Name. Small. ???
    Combine top5 and bottom five into 1 list. 
    '''
    
    #sort df by dependent variable then retrieve top five (x,y) pairs 
    #and indices
    sorted_df = df.sort_values(by=yname, ascending=ascending)
    top5_names = sorted_df.index[:5]
    top5_y = sorted_df[yname].values[:5]
    top5_x = sorted_df[xname].values[:5]
    
    #use markers and colors to make top 5 stand out
    marker = ['d','x','^','<','>']
    color_top = ['b', 'r', 'g','c','m']
    
    #create scatter plot of all points other than top 5 then label axes
    fig, ax = plt.subplots(1,1)    
    ax.scatter(sorted_df[xname].iloc[5:], sorted_df[yname].iloc[5:],color = 'k')
    ax.set_ylabel(yname)
    ax.set_xlabel(xname)
    #plot top 5 using designated markers and colors
    for name, y, x, marker, c in zip(top5_names, top5_y,top5_x, marker, color_top):
        ax.plot(x, y, marker,color=c, label =name)
        # names an be printed at points if this is uncommented
        #ax.annotate(name,(x,y),rotation=45+(np.random.rand()-0.5)*45,\
                    #fontsize='xx-small')
        
    #restrict plot window to 3 standard deviations above and below the mean
    #then label the axes using function input
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    #label for model predictions. Model is linear for defense and quadratic for offense.
    if len(model.params) == 2:
        label_string = f'EFG = {round(model.params[1],3)}*{xname}+{round(model.params[0],3)}'
    else: 
        label_string = f'EFG = {round(model.params[2],3)}*{xname}^2 + {round(model.params[1],3)}*{xname}+{round(model.params[0],3)}'
    
    #plot fitted values then save figure 
    ax.plot(df[xname], model.fittedvalues, label=label_string)
    ax.legend(framealpha = 0,fontsize = 'x-small', loc = 'best')
    ax.set_title(title)   
    fig.savefig(fname,bbox_inches='tight', pad_inches=0.5)