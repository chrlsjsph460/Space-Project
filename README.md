# Space-Project
Explore the effect of space to player's eFG


Installations: numpy pandas matplotlib.pyplot statsmodels.api statsmodels.formula.api sklearn.preprocessing statsmodels.graphics.gofplots.

Project Motivation: I enjoy playing basketball. It's hard to determine the effect of defense. For example, someone tall can cause you to alter your shot just by being near you. When I played pickup basketball in Cleveland, there was a law student, Nick, with very good shooting skills. It was very hard to make him miss. So, I was curious about the effect of space on professional basketball players. This is my humble attempt of getting some insights from NBA data using basic linear regression analysis.

File Description: shotLog_functions.py -> sequential_forward_selection: Automated model selection polyFit: Fit data using a polynomail function studentize: (X-mean)/sd fix_names: Put names in first name last name format lm_diagnostics: Show normal probability plot so we can assess normality of dataset lm_plot_rankx: given model, plot model and highlight according to position on x-axis lm_plot_ranky: given model, plot model and highlight according to position on x-axis

How to interact with your project: Mess around with the Jupyter notebooks.

Licensing, Authors, Acknowledgements, etc. Author: Charles Joseph Automated model selection: Kutner, M. H., Nachtsheim, C. J., & Neter, J. (2004). Applied Linear Regression Models: Michael H. Kutner, Christopher J. Nachtsheim, John Neter. McGraw-Hill.
