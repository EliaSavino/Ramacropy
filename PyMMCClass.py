import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os
import sif_parser as sp



class Spectra():
    def __init__(self, filepath='./DataFiles/Yourfile.csv', laser_wavelength=785.0):
        '''Initialize the Spectra object from a .sif file.

        Args:
            filepath (str): Path to the .sif file to open. Defaults to './DataFiles/Yourfile.csv'.
            laser_wavelength (float): Wavelength of the laser used in the measurement. Defaults to 785.0 nm.

        Raises:
            ValueError: If the file extension is not '.sif'.

        Returns:
            None
        '''
        # Check file extension
        if not filepath.endswith('.sif'):
            raise ValueError('Invalid file format. Must be .sif file.')

        # Load spectral data and information
        self.directory = filepath.split('/')
        self.SpectralData, self.SpectralInfo = sp.np_open(filepath)

        # Reshape spectral data for easier plotting
        self.SpectralData = self.SpectralData.transpose(2, 0, 1).reshape(self.SpectralData.shape[2], -1)

        # Extract Raman shift and timestamps from spectral information
        calib = sp.utils.extract_calibration(self.SpectralInfo)
        self.RamanShift = 1E7 * (1 / laser_wavelength - 1 / calib)
        self.TimeStamp = np.arange(0, self.SpectralInfo['CycleTime'] * self.SpectralData.shape[1],
                                   self.SpectralInfo['CycleTime'])

    def plot_kinetic(self):
        '''Plot the spectra for a kinetic run.

        Returns:
            None
        '''
        # Set up colormap and normalization
        cmap = cm.get_cmap('viridis_r')
        norm = colors.Normalize(vmin=self.TimeStamp.min(), vmax=self.TimeStamp.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        # Plot the lines with the color array
        fig, ax = plt.subplots()
        for i, t in enumerate(self.TimeStamp):
            ax.plot(self.RamanShift, self.SpectralData[:, i], c=sm.to_rgba(t))

        # Set axis labels and limits
        ax.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax.set_ylabel('Intensity (-)')
        ax.set_xlim(self.RamanShift.min(), self.RamanShift.max())
        ax.set_ylim(0.95 * self.SpectralData.min(), 1.05 * self.SpectralData.max())

        # Add a colorbar to the plot
        sm.set_array([])
        colorbar = plt.colorbar(sm)
        colorbar.set_label('Time (s)')

        # Show the plot
        plt.show()

    def plot_few(self, other_spectra=[], labels=[]):
        '''Plot a single or multiple spectra for comparison.

        Args:
            other_spectra (list): List of other Spectra instances to plot.
            labels (list): List of labels for the spectra. If not provided, filenames are used.

        Raises:
            ValueError: If any of the spectra has more than one column.

        Returns:
            None
        '''
        # Check for single spectrum
        if self.SpectralData.shape[1] != 1:
            raise ValueError('This is not a single spectrum. Use plot_kinetic function instead.')

        # Initialize plot and axis settings
        fig, ax = plt.subplots()
        ax.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax.set_ylabel('Intensity (-)')
        ax.set_xlim(self.RamanShift.min(), self.RamanShift.max())

        # Set up colormap
        num_spectra = len(other_spectra) + 1
        colors = cm.jet(np.linspace(0, 1, num_spectra))

        # Plot spectra
        ax.plot(self.RamanShift, self.SpectralData, c=colors[0])
        for i, spec in enumerate(other_spectra):
            if spec.SpectralData.shape[1] != 1:
                raise ValueError('One of the spectra is not a single spectrum. Use PlotKinetic function instead.')
            ax.plot(spec.RamanShift, spec.SpectralData, c=colors[i + 1])

        # Set legend labels
        if len(labels) != num_spectra:
            labels = labels+[spec.directory[-1].replace('.sif', '') for spec in other_spectra[len(labels)-1:]]
            if len(labels) != num_spectra:
                raise ValueError('You gave too many labels. Try again.')

        # Set legend and show plot
        ax.legend(labels)
        plt.show()


