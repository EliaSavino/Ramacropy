import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.widgets import Slider,Button
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import savgol_filter
import sif_parser as sp


def bline(x,ydat, coarsness=0.0, angle=0.0, offset=0.0):
    '''
    Generates the ndarray of length equal to that of the ydata that will be used as baseline.

    Args:
        ydat (ndarray): input ydata array
        coarsness (float): coarseness of the baseline (default=0.0)
        angle (float): angle of rotation in degrees (default=0.0)
        offset (float): offset of the baseline (default=0.0)

    Returns:
        ndarray: ydata of baseline
    '''
    # Generate virtual x array
    # virtual_x = np.arange(len(ydat))
    Yline = np.zeros(len(x))
    virtual_x = x

    Yline += offset
    Yline = (1 - coarsness) * Yline + coarsness * ydat
    window = int((1-coarsness)*700)+1 if int((1-coarsness)*700) % 2 == 0 else int((1-coarsness)*700)
    Yline = savgol_filter(Yline, window_length = window, polyorder = 2)
    # Rotate the ydata around its center
    center_x, center_y = np.mean(virtual_x), np.mean(Yline)
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    translated_coords = np.vstack((virtual_x - center_x, Yline - center_y))
    rotated_coords = np.dot(rotation_matrix, translated_coords)
    Yline = rotated_coords[1, :] + center_y

    # Apply offset and coarseness


    return Yline

def InteractiveBline(RamanShift, SpectralData):
    # Define initial parameters
    init_coarsness = 0
    init_angle = 0
    init_offset = 0
    x = np.arange(len(SpectralData))

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, bottom=0.25)
    raw, = ax.plot(RamanShift, SpectralData[:, 0], c='r')
    baseline, = ax.plot(RamanShift, bline(RamanShift, SpectralData[:, 0], init_coarsness, init_angle, init_offset), lw=2)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_xlim(RamanShift.min(),RamanShift.max())
    ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())

    # Add sliders for coarseness, angle, and offset
    axcoarse = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    coarse_slider = Slider(ax=axcoarse, label='Coarsness', valmin=0.0, valmax=1.0, valinit=init_coarsness)
    axang = fig.add_axes([0.12, 0.25, 0.0225, 0.63])
    ang_slider = Slider(ax=axang, label='Angle', valmin=-90, valmax=90, valinit=init_angle, orientation='vertical')
    axoff = fig.add_axes([0.05, 0.25, 0.0225, 0.63])
    off_slider = Slider(ax=axoff, label='Offset', valmin=-0.1 * SpectralData[:, 0].max(), valmax=0.1 * SpectralData[:, 0].max(), valinit=init_offset, orientation='vertical')

    # Define a function to update the baseline when sliders are changed
    def update(val):
        baseline.set_ydata(bline(x, SpectralData[:, 0], coarse_slider.val, ang_slider.val, off_slider.val))
        fig.canvas.draw_idle()

    # Connect the update function to the slider events
    coarse_slider.on_changed(update)
    ang_slider.on_changed(update)
    off_slider.on_changed(update)

    # Add buttons to reset, apply, and save the baseline values
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    ResetButton = Button(resetax, 'Reset', hovercolor='0.975')
    tryax = fig.add_axes([0.69, 0.025, 0.1, 0.04])
    TryButton = Button(tryax, 'Try', hovercolor='0.975')
    doneax = fig.add_axes([0.58, 0.025, 0.1, 0.04])
    DoneButton = Button(doneax, 'Done', hovercolor='0.975')

    # Define functions for button callbacks
    def reset(event):
        raw.set_ydata(SpectralData[:, 0])
        coarse_slider.reset()
        ang_slider.reset()
        off_slider.reset()
        fig.canvas.draw_idle()

    def apply_bline(event):
        y_corrected = SpectralData[:, 0] - baseline.get_ydata()
        raw.set_ydata(y_corrected)
        fig.canvas.draw_idle()

    def save_vals(event):
        plt.close()
        return ang_slider.val, coarse_slider.val, off_slider.val

    # Connect the button callbacks to the button events
    ResetButton.on_clicked(reset)
    TryButton.on_clicked(apply_bline)
    DoneButton.on_clicked(save_vals)
    plt.show()
    return ang_slider.val, coarse_slider.val, off_slider.val





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

