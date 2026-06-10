"""
🧮 Análise no Espaço de Estados
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import expm
from scipy import signal
import control
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings


def run():

    warnings.filterwarnings("ignore")

    # st.set_page_config(...)  ← remover se já existir no app principal

    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "lines.linewidth": 1.6,
        "font.family": "serif",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "grid.alpha": 0.3,
    })

    # TODO: todo o restante do código original
    # exatamente como está no arquivo
