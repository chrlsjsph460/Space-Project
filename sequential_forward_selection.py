import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from itertools import combinations as combo
import patsy
from patsy import (ModelDesc, EvalEnvironment, Term, 
                   EvalFactor, LookupFactor, demo_data, dmatrix)


    
def sequential_forward_selection(data, predictors, response, max_length = 10, numerical_predictors = [], entry_limit = 0.10):
    """
    Author: Charles Joseph 6/13/2022. 
    Maybe something like this is out there. If it is, I couldn't find it
    Get all possible 2nd order model.
    Initialize model to only have intercept.
    Subsequently add predictors one at a time. 
    Add predictor with the smallest p-value that improves the adjusted r2
    Stop when maximum length is attained or pvalue is larger than entry threshold.
    
    
    Inputs: data         - pandas df of response and predictor (pd.DataFrame)
            predictors   - list of predictor column names (string)
            response     - response as a string (string)
            max_length - maximum model length
            entry_limit  - Only add predictors st. pval < entry_limit
            
            output     - smf.ols model. Model is chosen as described above
            
    Scoring: Adjusted R Squared
    
    
    """
    
    
    ##########################################################################
    ################### BUILD LIST OF PREDICTORS  ############################
    ##########################################################################
    
    # build list of predictors for second order model.
    # numerical predictors can be squared
    
    
    full_rhs = []
    full_rhs += [Term([LookupFactor(x)]) for x in predictors] #add predictors
    full_rhs += [Term([EvalFactor("np.power("+name+", 2)")]) for name in predictors if name in numerical_predictors] # add squares
    full_rhs += [Term([LookupFactor(x[0]), LookupFactor(x[1])]) for x in list(combo(predictors,2))] #add interactions
    
    # lhs is the response variable
    lhs = [Term([LookupFactor(response)])]
    
    
    ###########################################################################
    ################# INITIALIZE MODEL ########################################
    ###########################################################################
    # initialize model with no predictors.
    
    rhs_candidate = [patsy.INTERCEPT]
    desc = ModelDesc(lhs, rhs_candidate)
    model = smf.ols(desc.describe(), data = data).fit()
    
    
    # only add a factor if p-value is less than entry limit
    # remove a factor if p-value rises to above release_limit
    
    potential_rhs = full_rhs.copy()
    rsquared_adj = 0.0
    new_p = 0.005
    #while we have  potential factors in our list
    #while the last add on was an improvement 
    
    ###########################################################################
    ################ ITERATE THROUGH LIST TO FIND A GOOD MODEL ################
    ###########################################################################
    
    num_predictors = 0
    while potential_rhs and new_p < entry_limit and num_predictors < max_length:
        # list of tuples (r2, term, p)
        num_predictors += 1
        print(f'Iteration {num_predictors}')
        rsquared_and_term = []
        current_params = set(model.params.index.tolist())
        
        for term in potential_rhs:        
            new_rhs_candidate = rhs_candidate.copy() + [term] #Proposed model = old model + new factor
            desc = ModelDesc(lhs, new_rhs_candidate)
            model = smf.ols(desc.describe(), data = data).fit() #fit model
            
            #get model parameters that were added. It's necessary to use sets because
            #more than one parameter is added if it's categorical
            #and the order the parameters are listed doesn't stay the same
            candidate_params = set(model.params.index.tolist())
            new_params = candidate_params.difference(current_params)
            
            #get pvalues for new parameters
            proposed_p = min([model.pvalues[param] for param in new_params])
            
            
        
            #if new model is improvement and factor is significant add to tuple
            if model.rsquared_adj > rsquared_adj and proposed_p < entry_limit:
                rsquared_and_term.append((model.rsquared_adj, term, proposed_p))
            
            
        if len(rsquared_and_term) > 0:
            rsquared_and_term.sort(key = lambda x: x[2]) #sort by pvalue
            rhs_candidate = rhs_candidate + [rsquared_and_term[0][1]] #add term with smallest pvalue
            potential_rhs.remove(rsquared_and_term[0][1])
            new_p = rsquared_and_term[0][2]
            # fit chosen model
            desc = ModelDesc(lhs, rhs_candidate)
            model = smf.ols(desc.describe(), data = data).fit()
        else:
            new_p = 1.0
        
        
        desc = ModelDesc(lhs, rhs_candidate)
        model = smf.ols(desc.describe(), data = data).fit()    
        print(f'The new p-value is {new_p} and the new rsquared_adj is {model.rsquared_adj}')   
        
    return model

'''        
# test data 
url = "http://data.princeton.edu/wws509/datasets/salary.dat"
data = pd.read_csv(url, sep='\\s+')
print(data.head())



predictors = list(data.columns).copy()
predictors.remove('sl')
response = 'sl'   
model_ = sequential_forward_selection(data, predictors, response, 3,\
                             numerical_predictors = ['yr','yd'], entry_limit = 0.05)
'''