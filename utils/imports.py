# Basics
import pandas as pd
import numpy as np
import random
import operator

# Stats
import scipy.stats as scs
from scipy.stats import norm

# Vizualisation basics
import matplotlib.pyplot as plt
import matplotlib.colors as clr
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style("whitegrid")

# Matplotlyb 3D
from mpl_toolkits import mplot3d

# Colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Vizualisation Plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot # plot offline
init_notebook_mode(connected=True)

# Widgets
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual
import ipywidgets as widgets
from IPython.display import display