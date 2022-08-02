# Space-Project
Explore the effect of space to player's effective field goal percentage (eFG)

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting-started)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgments](#acknowledgements)
6. [Notebook](#notebook)

## Description
Project Motivation: I enjoy playing basketball. It's hard to determine the effect of defense. For example, someone tall can cause you to alter your shot just by being near you. When I played pickup basketball in Cleveland, there was a law student, Nick, with very good shooting skills. It was very hard to make him miss. So, I was curious about the effect of space on professional basketball players. This is my humble attempt of getting some insights from NBA data using basic linear regression analysis.

File Description: shotLog_functions.py -> sequential_forward_selection: Automated model selection polyFit: Fit data using a polynomail function studentize: (X-mean)/sd fix_names: Put names in first name last name format lm_diagnostics: Show normal probability plot so we can assess normality of dataset lm_plot_rankx: given model, plot model and highlight according to position on x-axis lm_plot_ranky: given model, plot model and highlight according to position on x-axis

## Getting Started
### Dependencies
- Python 3
  - Jupyter Notebook
  - ETL Library: 
    - Pandas 
    - Numpy
  - Linear Regression: 
    - statsmodels 
    - sklearn 
  - Plots:
    - matplotlib
    - seaborn

## Authors
- Charles Joseph

## License


## Acknowledgements
- Kutner, M. H., Nachtsheim, C. J., & Neter, J. (2004). Applied Linear Regression Models: Michael H. Kutner, Christopher J. Nachtsheim, John Neter. McGraw-Hill. for their section on automatic model selection
- [kaggle](https://www.kaggle.com/) for their NBA shotlog dataset

## Notebooks
[automatic model selection]()
[other helper functions]()
[main file]()
[html of main file]()

How to interact with your project: Mess around with the Jupyter notebooks.

