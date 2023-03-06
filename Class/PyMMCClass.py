import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sif_parser as sp



class Spectra():
    def __init__(self, Filepath = '.\DataFiles\Yourfile.csv', ):
        '''The initialisation of this class requires a relative path to file (put your data in the Datafile dir)'''
        self.directory = str.split(Filepath,)
        self.SpectralData = pd.read_csv(Filename)

