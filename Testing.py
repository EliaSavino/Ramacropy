import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.widgets import Slider,Button
import numpy as np
import os
import sif_parser as sp
import PyMMCClass
from scipy.signal import savgol_filter
#
Spec = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_kinetic.sif')
Spec.plot_kinetic()
Spec.baseline(Interactive=True)
Spec.plot_kinetic()
# # Spec2 = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_before.sif')
# # Spec.plot_few(other_spectra=[Spec2,Spec2,Spec],labels=['After','Before'])
#
# Spec3 = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_kinetic.sif')
# # Spec3.plot_kinetic()
# # print(type(Spec3.SpectralData))
# for object in Spec.SpectralData.T:
#     print(object)

# The parametrized function to be plott






