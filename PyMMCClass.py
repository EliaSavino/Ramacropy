
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import sif_parser as sp
from Utils import *



class Spectra():
    def __init__(self, filepath='./DataFiles/Yourfile.csv', laser_wavelength=785.0):
        '''Initialize the Spectra object from a .sif, .pkl or .csv file. (pkl or csv files only if generated from this script)

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
        self.filelab = self.directory[-1][:-4]
        self.SpectralData, self.SpectralInfo = sp.np_open(filepath)

        # Reshape spectral data for easier plotting
        self.SpectralData = self.SpectralData.transpose(2, 0, 1).reshape(self.SpectralData.shape[2], -1)

        # Extract Raman shift and timestamps from spectral information
        calib = sp.utils.extract_calibration(self.SpectralInfo)
        self.RamanShift = 1E7 * (1 / laser_wavelength - 1 / calib)
        self.TimeStamp = np.arange(0, self.SpectralInfo['CycleTime'] * self.SpectralData.shape[1],
                                   self.SpectralInfo['CycleTime'])
        # self.UID = GenID()

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
        ax.plot(self.RamanShift, self.SpectralData, c=colors[0], label = self.filelab)
        for i, spec in enumerate(other_spectra):
            if spec.SpectralData.shape[1] != 1:
                raise ValueError('One of the spectra is not a single spectrum. Use PlotKinetic function instead.')
            ax.plot(spec.RamanShift, spec.SpectralData, c=colors[i + 1], label = spec.filelab)

        hand, lab = ax.get_legend_handles_labels()

        if len(labels) > len(lab):
            print('you gave too many labels, your input is ignored!')
        else:
            for i in range(len(labels)):
                lab[i] = labels[i]
        # Set legend and show plot
        ax.legend(hand, lab)
        plt.show()

    def baseline(self, coarsness=0.0, angle=0.0, offset=0.0, interactive=False):
        """
        Corrects the baseline of your spectra. (if the spectra are kinetic uses individual baselines)
        Remember, the baseline brings the spectra to it (i.e. spectra > baseline decreases, baseline > spectra increases)

        Args:
            coarsness (float): Level of similarity between the spectra and the baseline, at 0.0 the baseline is straight,
                at 1.0 the baseline is the same as the spectrum. (lower is better)
            angle (float): What is the angle of the baseline, from -90 to 90. Use this with care.
            offset (float): How high up is the baseline.
            interactive (bool): Whether to show an interactive plot to adjust the baseline parameters (default: False)

        Raises:
            ValueError: If the arguments are out of bounds.

        Returns:
            None
        """
        if coarsness < 0.0 or coarsness > 1.0 or abs(angle) > 90.0:
            raise ValueError("One of your arguments is out of bounds! Try again.")

        if interactive:
            angle, coarsness, offset = InteractiveBline(self.RamanShift, self.SpectralData)
        elif coarsness == 0.0 and angle == 0.0 and offset == 0.0:
            print("Your baseline is all 0s, quit messing around and do something.")
            return

        for count in range(self.SpectralData.shape[1]):
            self.SpectralData[:, count] -= bline(self.RamanShift, self.SpectralData[:, count], coarsness, angle, offset)

    def normalise(self, method='area', interactive=False, **kwargs):
        '''
        normalises the spectra either by the peak or area.
        args:
         method (string): method can be either 'area' or 'peak' the first normalises by area and the second normalises by peak
         interactive (bool): either True or False, if true opens a window in the selected method that you can use to
                            figure out where the bounds of normalisation are
        kwargs: optional (required if interactive is False) keyword arguments to decide the bounds of normalisation
                if method is 'area': the keyword arguments are: start = float and end = float
                if method is 'peak': only one keyword argument is use: peak = float

        :return: nothing
        '''
        if method.lower() not in ['area', 'peak']:
            raise ValueError('Not recognised type of method, either: area or peak')

        if interactive:
            if method.lower() == 'area':
                bounds = InteractiveNormaliseArea(self.RamanShift, self.SpectralData)
                start_pos, end_pos = np.abs(self.RamanShift - bounds[0]).argmin(), np.abs(
                    self.RamanShift - bounds[1]).argmin()
            else:
                peak_pos = InteractiveNormalisePeak(self.RamanShift, self.SpectralData)
        else:
            if method.lower() == 'area':
                try:
                    bounds = sorted([kwargs['start'], kwargs['end']])
                except KeyError:
                    print('You must have used the wrong keywords, use start and end')
                    return

                if (self.RamanShift.min()<= bounds[1]<= self.RamanShift.max()) and (self.RamanShift.min()<= bounds[1]<= self.RamanShift.max()):
                    start_pos, end_pos = np.abs(self.RamanShift - bounds[0]).argmin(), np.abs(self.RamanShift - bounds[1]).argmin()
                else:
                    raise ValueError('Your chosen start and end area values are out of bounds.')
            else:
                try:
                    peak_pos = kwargs['peak']
                except KeyError:
                    print('You must have used the wrong keyword, use peak')
                    return
                if peak_pos < self.RamanShift.min() or peak_pos > self.RamanShift.max():
                    raise ValueError('The chosen peak position is out of bounds.')
                else:
                    peak_pos = np.abs(self.RamanShift - peak_pos).argmin()

        if method.lower() == 'area':
            for count in range(self.SpectralData.shape[1]):
                self.SpectralData[:, count] = normalise_area(self.SpectralData[:, count], start_pos, end_pos)
        else:
            for count in range(self.SpectralData.shape[1]):
                self.SpectralData[:, count] = normalise_peak(self.SpectralData[:, count], peak_pos)

    def integrate(self, start = 0.0, end = 0.0, interactive = False):
        '''
        integrates the spectrum/spectra and makes a new property of the Specra class called .integral that you can access.
        (works on both kinetic or single spectra)

        :param start: Starting shift of integration
        :param end: ending shift of integration
        :param interactive: Shows plot so you can find this manually
        :return: None
        '''
        if interactive:
            bounds = InteractiveIntegrateArea(self.RamanShift,self.SpectralData)
        else:
            bounds = sorted([start,end])

        if not((self.RamanShift.min() <= bounds[0] <= self.RamanShift.max()) and (self.RamanShift.min() <= bounds[1]<= self.RamanShift.max())):
            raise ValueError('Your chosen start and end values are out of bounds!')

        start_pos,end_pos = np.abs(self.RamanShift - bounds[0]).argmin(), np.abs(self.RamanShift - bounds[1]).argmin()

        self.integral = []
        for count in range(self.SpectralData.shape[1]):
            self.integral.append(integrate_area(self.SpectralData[:,count],start_pos, end_pos))

    def plot_itegral_kinetic(self,other_spectra = [], labels = [], conversion = False):
        '''
        Plots a trace of integral over time, or conversion over time. you can optionally add multiple instances of Spectra class
        (integration must have been performed on them) to plot and compare multiple traces. It is suggested to normalise all on the same band

        :param other_spec list of obj: optional other instances of the Spectra class
        :param labels list of str: list of labels to name your traces, if not present uses filenames
        :param conversion bool: If false plots integral, if true plots conversion (calc as 1-I/I0)

        :return: none
        '''
        if self.SpectralData.shape[1] == 1:
            raise ValueError('This is a single spectrum, not really worth it to plot the integral like this'
                             ', better off using the approrpiate function')
        if not(hasattr(self,'integral')):
            raise AttributeError('You need to integrate first before trying to plot it.')

        fig, ax = plt.subplots()
        ax.set_xlabel('Time (s)')

        # Set up colormap
        num_spectra = len(other_spectra) + 1
        colors = cm.jet(np.linspace(0, 1, num_spectra))

        if conversion:
            ax.set_ylabel('Conversion')
            ax.set_ylim(0,1)
            ax.scatter(self.TimeStamp,1 - self.integral/self.integral[0], color = colors[0], label = self.filelab)
        else:
            ax.set_ylabel('Integral')
            ax.scatter(self.TimeStamp,self.integral, color = colors[0], label = self.filelab)

        for i, spec in enumerate(other_spectra):
            if spec.SpectralData.shape[1] == 1:
                raise ValueError('This is a single spectrum, not really worth it to plot the integral like this'
                                 ', better off using the approrpiate function')
            if not (hasattr(spec, 'integral')):
                raise AttributeError('You need to integrate first before trying to plot it.')

            if conversion:
                ax.scatter(spec.TimeStamp,1-spec.integral/spec.integral[0], color = colors[i],label = spec.filelab)
            else:
                ax.scatter(spec.TimeStamp,spec.integral, color = colors[i], label = spec.filelab)

        hand,lab = ax.get_legend_handles_labels()

        if len(labels)>len(lab):
            print('you gave too many labels, your input is ignored!')
        else:
            for i in range(len(labels)):
                lab[i] = labels[i]
        # Set legend and show plot
        ax.legend(hand,lab)
        ax.set_xlim(0, None)
        plt.show()

    def plot_integral_single(self,other_spectra = [], labels = []):
        '''
        Plots a trace of integral over spectra label. you can optionally add multiple instances of Spectra class
        (integration must have been performed on them) to plot and compare multiple integrals. It is suggested to normalise all on the same band

        :param other_spec list of obj: optional other instances of the Spectra class
        :param labels list of str: list of labels to name your traces, if not present uses filenames

        :return: none
        '''

        if self.SpectralData.shape[1] != 1:
            raise ValueError('This is a kinetic spectrum, not really worth it to plot the integral like this'
                             ', better off using the approrpiate function')
        if not(hasattr(self,'integral')):
            raise AttributeError('You need to integrate first before trying to plot it.')

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom = 0.2)
        ax.set_ylabel('Integral')


        # Set up colormap
        num_spectra = len(other_spectra) + 1
        colors = cm.jet(np.linspace(0, 1, num_spectra))

        x = np.arange(num_spectra)
        lab = [self.filelab]

        ax.scatter(x[0], self.integral, color = colors[0])

        for i, spec in enumerate(other_spectra):
            if spec.SpectralData.shape[1] != 1:
                raise ValueError('This is a kinetic spectrum, not really worth it to plot the integral like this'
                                 ', better off using the approrpiate function')
            if not (hasattr(spec, 'integral')):
                raise AttributeError('You need to integrate first before trying to plot it.')


            ax.scatter(x[i+1], spec.integral, color=colors[i])
            lab.append(spec.filelab)


        if len(labels) > len(lab):
            print('you gave too many labels, your input is ignored!')
        else:
            for i in range(len(labels)):
                lab[i] = labels[i]
        # Set legend and show plot
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation = 35)

        plt.show()

# def plot_integral_kinetic(self):
    #     '''temporarily empty'''
    #
    # def save_changes(self, dirpath = self.directory[:-1], filename = self.directory[-1]):
    #     '''temporarily empty'''






