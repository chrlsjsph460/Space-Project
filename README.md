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
Project Motivation: I enjoy playing basketball. It's hard to determine the effect of defense. For example, someone tall can cause you to alter your shot just by being near you. When I played pickup basketball in Cleveland, there was a law student, Nick, with excellent shooting skills. It was very hard to make him miss. You would have to really get in his space to throw him off. So, I was curious about the effect of space on professional basketball players. This is my humble attempt of getting some insights from NBA data using basic linear regression analysis.

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
[MIT OPENSOURCE LICENSE](LICENSE.TXT)

## Acknowledgements
- Kutner, M. H., Nachtsheim, C. J., & Neter, J. (2004). Applied Linear Regression Models: Michael H. Kutner, Christopher J. Nachtsheim, John Neter. McGraw-Hill. for their section on automatic model selection
- [Kaggle](https://www.kaggle.com/) for their [NBA shot logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs) dataset

## Notebooks
- [Automatic Model Selection](sequential_forward_selection.ipynb) -> Notebook with automated model selection function
- [Plotting functions](shotLog_functions.ipynb) -> Notebook of various plotting functions for project
- [Main File](CrispdmProject.ipynb) -> Notebook where model selection and plotting are used. There's some discussion. 
- [Html of project](Space-Project.html) -> Html version of main file

How to interact with this project: download notebooks above and run all cells or download and view html of project without running anything. 
