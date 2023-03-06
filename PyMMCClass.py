import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os
import sif_parser as sp



class Spectra():
    def __init__(self, Filepath = './DataFiles/Yourfile.csv', LaserWavelength = 785.0):
        '''The initialisation of this class requires a relative path to file (put your data in the Datafile dir)
        .sif files only'''
        self.directory = str.split(Filepath,'/') #Here stores the filepath in pieces
        self.SpectralData, self.SpectralInfo = sp.np_open(Filepath) # here it parses the data
        self.SpectralData = self.SpectralData.transpose(2,0,1).reshape(self.SpectralData.shape[2],-1)# this fixes the shape of the array so it's easyer to plot
        self.RamanShift = 1E7*(1/LaserWavelength - 1/sp.utils.extract_calibration(self.SpectralInfo))#extracts the calibration and calculates the shift
        self.TimeStamp = np.arange(0,self.SpectralInfo['CycleTime']*self.SpectralData.shape[1],self.SpectralInfo['CycleTime'])


    def PlotKinetic(self):
        '''This plots the spectra for a kinetic run, either use this as a prehemptive data visualisation or the final plotting tool'''
        cmap = cm.get_cmap('viridis_r')

        # Create a normalize object to map timestamps to colors
        normalize = colors.Normalize(vmin=self.TimeStamp.min(), vmax=self.TimeStamp.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
        # Generate the color array by applying the normalize function to the timestamps
        # Plot the lines with the color array
        fig, ax = plt.subplots()
        # # ax.set_prop_cycle('color', colors)
        for index, time in enumerate(self.TimeStamp):
            # print(index,time)
            ax.plot(self.RamanShift, self.SpectralData[:, index], c=sm.to_rgba(time))
        ax.set_xlabel('Raman Shift (cm$^{-1}$')
        ax.set_ylabel('Intensity (-)')
        ax.set_xlim(self.RamanShift.min(),self.RamanShift.max())
        ax.set_ylim(0.95*self.SpectralData.min(), 1.05*self.SpectralData.max())
        # # Add a colorbar to the plot
        #
        sm.set_array([])
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Time (s)')


        # Show the plot
        plt.show()

    def PlotFew(self, OtherSpectra = [], Labels = []):
        '''Here you can either plot a single spectrum (from a non kinetic run) or plot a few spectra together to compare them
         for best results use this for prehemptive data visualisation with the raw spectra. When comparing spectra it is best to use
         spectra that already have been processed (i.e. baselined and normalised),you can pass other Spectra classes to this to plot multiple
        you should always pass at least 1 label to this function'''
        if self.SpectralData.shape[1]>1:
            raise Exception('This is not a single spectrum, i guess it is a kinetic run, you should use the PlotKinetic function instead')
            return
        fig,ax = plt.subplots()
        ax.set_xlabel('Raman Shift (cm$^{-1}$')
        ax.set_ylabel('Intensity (-)')
        ax.set_xlim(self.RamanShift.min(),self.RamanShift.max())
        ax.set_ylim(0.95*self.SpectralData.min(), 1.05*self.SpectralData.max())
        jet_colors = cm.jet(np.linspace(0,1,len(OtherSpectra)+1))
        ax.plot(self.RamanShift,self.SpectralData)

        ax.plot()


        if len(OtherSpectra) == 0:

