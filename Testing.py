import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os
import sif_parser as sp
import PyMMCClass

Spec = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_after.sif')
Spec2 = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_before.sif')
Spec.plot_few(other_spectra=[Spec2,Spec2,Spec],labels=['After','Before'])

Spec3 = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_kinetic.sif')
Spec3.plot_kinetic()