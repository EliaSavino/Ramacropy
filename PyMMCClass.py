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

def integrate_area(y_dat, start_pos, stop_pos):
    """Integrate the area under a numpy array between two positions.

    Args:
        y_dat (numpy.ndarray): The array to integrate.
        start_pos (int): The starting position for integration.
        stop_pos (int): The stopping position for integration.

    Returns:
        float: The area under the array between the start and stop positions.
    """
    x = np.arange(len(y_dat))
    mask = (x >= start_pos) & (x <= stop_pos)
    area = np.trapz(y_dat[mask], x[mask])
    return area

def normalise_peak(y_dat,peak_pos):
    """Normalize a numpy array by dividing all values by the value at a specified position.

    Args:
        y_dat (numpy.ndarray): The array to normalize.
        peak_pos (int): The position to use for normalization.

    Returns:
        numpy.ndarray: The normalized array.
    """

    norm_factor = y_dat[peak_pos]
    normalized_data = y_dat/norm_factor
    return normalized_data

def normalise_area(y_dat,start_pos,end_pos):
    """Normalize the area under a curve between two specified positions.

        Args:
            y_dat (numpy.ndarray): The array containing the y values of the curve.
            start_pos (float): The starting position for the normalization.
            end_pos (float): The ending position for the normalization.

        Returns:
            numpy.ndarray: The normalized curve.
        """

    norm_factor = integrate_area(y_dat,start_pos,end_pos)
    normalized_data = y_dat/norm_factor
    return normalized_data

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

def InteractiveNormalisePeak(RamanShift, SpectralData):
    # Define initial parameters
    init_peak = RamanShift.min()
    x = np.arange(len(SpectralData))

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.25)
    raw, = ax.plot(RamanShift, SpectralData[:, 0], c='r')
    line = ax.axvline(x = init_peak,lw = 0.5)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_xlim(RamanShift.min(),RamanShift.max())
    ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())

    # Add sliders for coarseness, angle, and offset
    axpeak = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    peak_slider = Slider(ax=axpeak, label='Peak Position', valmin=RamanShift.min(), valmax=RamanShift.max(), valinit=init_peak)


    # Define a function to update the baseline when sliders are changed
    def update(val):
        line.set_xdata(peak_slider.val)
        fig.canvas.draw_idle()

    # Connect the update function to the slider events
    peak_slider.on_changed(update)


    # Add buttons to reset, apply, and save the baseline values
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    ResetButton = Button(resetax, 'Reset', hovercolor='0.975')
    tryax = fig.add_axes([0.69, 0.025, 0.1, 0.04])
    TryButton = Button(tryax, 'Try', hovercolor='0.975')
    doneax = fig.add_axes([0.58, 0.025, 0.1, 0.04])
    DoneButton = Button(doneax, 'Done', hovercolor='0.975')

    # Define functions for button callbacks
    def reset(event):
        line.set_xdata(init_peak)
        raw.set_ydata(SpectralData[:,0])
        peak_slider.reset()
        ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())
        fig.canvas.draw_idle()


    def apply_norm(event):
        peak_pos = np.abs(RamanShift-peak_slider.val).argmin()
        y_corrected = normalise_peak(SpectralData[:,0],peak_pos)
        raw.set_ydata(y_corrected)
        ax.set_ylim(0.95*y_corrected.min(),1.05*y_corrected.max())
        fig.canvas.draw_idle()

    def save_vals(event):
        plt.close()
        return peak_slider

    # Connect the button callbacks to the button events
    ResetButton.on_clicked(reset)
    TryButton.on_clicked(apply_norm)
    DoneButton.on_clicked(save_vals)
    plt.show()
    return peak_slider

def InteractiveNormaliseArea(RamanShift, SpectralData):
    # Define initial parameters
    init_start = RamanShift.min()
    init_end = RamanShift.max()
    y1 = np.zeros(len(RamanShift))

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.3)
    raw, = ax.plot(RamanShift, SpectralData[:, 0], c='r')
    fill = ax.fill_between(RamanShift,y1,SpectralData[:,0], where=((RamanShift >= init_start) & (RamanShift <= init_end)),
                           color='purple', alpha=0.3)
    start = ax.axvline(x = init_start,lw = 0.5)
    end = ax.axvline(x = init_end, lw = 0.5)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_xlim(RamanShift.min(),RamanShift.max())
    ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())

    # Add sliders for coarseness, angle, and offset
    axstart = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    start_slider = Slider(ax=axstart, label='Peak Position', valmin=RamanShift.min(), valmax=RamanShift.max(), valinit=init_start)
    axend = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    end_slider = Slider(ax=axend, label='Peak Position', valmin=RamanShift.min(), valmax=RamanShift.max(),
                         valinit=init_end)


    # Define a function to update the baseline when sliders are changed
    def update(val):
        if start_slider.val<end_slider.val:
            start.set_xdata(start_slider.val)
            end.set_xdata(end_slider.val)
            dummy = ax.fill_between(RamanShift,y1,raw.get_ydata(),
                                    where=((RamanShift >=start_slider.val) & (RamanShift <= end_slider.val)),alpha = 0)
        else:
            end.set_xdata(start_slider.val)
            start.set_xdata(end_slider.val)
            dummy = ax.fill_between(RamanShift, y1, raw.get_ydata(),
                                    where=((RamanShift >= end_slider.val) & (RamanShift <= start_slider.val)), alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])

        fig.canvas.draw_idle()

    # Connect the update function to the slider events
    start_slider.on_changed(update)
    end_slider.on_changed(update)


    # Add buttons to reset, apply, and save the baseline values
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    ResetButton = Button(resetax, 'Reset', hovercolor='0.975')
    tryax = fig.add_axes([0.69, 0.025, 0.1, 0.04])
    TryButton = Button(tryax, 'Try', hovercolor='0.975')
    doneax = fig.add_axes([0.58, 0.025, 0.1, 0.04])
    DoneButton = Button(doneax, 'Done', hovercolor='0.975')

    # Define functions for button callbacks
    def reset(event):
        dummy = ax.fill_between(RamanShift, y1, SpectralData[:, 0],
                                where=((RamanShift >= init_start) & (RamanShift <= init_end)), alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])
        start.set_xdata(init_start)
        end.set_xdata(init_end)
        raw.set_ydata(SpectralData[:,0])
        start_slider.reset()
        end_slider.reset()
        ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())
        fig.canvas.draw_idle()



    def apply_norm(event):
        positions = (np.abs(RamanShift-start_slider.val).argmin(),np.abs(RamanShift-end_slider.val).argmin())
        position = sorted(positions)
        y2 = PyMMCClass.normalise_area(SpectralData[:,0],*position)

        raw.set_ydata(y2)
        ax.set_ylim(0.95*y2.min(),1.05*y2.max())
        dummy = ax.fill_between(RamanShift, y1, raw.get_ydata(),
                                where=((RamanShift >= RamanShift[position[0]]) & (RamanShift <= RamanShift[position[1]])), alpha = 0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])
        fig.canvas.draw_idle()

    def save_vals(event):
        plt.close()
        bounds = (start_slider.val,end_slider.val)
        return sorted(bounds)

    # Connect the button callbacks to the button events
    ResetButton.on_clicked(reset)
    TryButton.on_clicked(apply_norm)
    DoneButton.on_clicked(save_vals)
    plt.show()
    bounds = (start_slider.val, end_slider.val)
    return sorted(bounds)

def InteractiveIntegrateArea(RamanShift, SpectralData):
    # Define initial parameters
    init_start = RamanShift.min()
    init_end = RamanShift.max()
    y1 = np.zeros(len(RamanShift))

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.3)
    raw, = ax.plot(RamanShift, SpectralData[:, 0], c='r')
    fill = ax.fill_between(RamanShift,y1,SpectralData[:,0], where=((RamanShift >= init_start) & (RamanShift <= init_end)),
                           color='purple', alpha=0.3)
    start = ax.axvline(x = init_start,lw = 0.5)
    end = ax.axvline(x = init_end, lw = 0.5)
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_xlim(RamanShift.min(),RamanShift.max())
    ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())

    # Add sliders for coarseness, angle, and offset
    axstart = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    start_slider = Slider(ax=axstart, label='Peak Position', valmin=RamanShift.min(), valmax=RamanShift.max(), valinit=init_start)
    axend = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    end_slider = Slider(ax=axend, label='Peak Position', valmin=RamanShift.min(), valmax=RamanShift.max(),
                         valinit=init_end)


    # Define a function to update the baseline when sliders are changed
    def update(val):
        if start_slider.val<end_slider.val:
            start.set_xdata(start_slider.val)
            end.set_xdata(end_slider.val)
            dummy = ax.fill_between(RamanShift,y1,raw.get_ydata(),
                                    where=((RamanShift >=start_slider.val) & (RamanShift <= end_slider.val)),alpha = 0)
        else:
            end.set_xdata(start_slider.val)
            start.set_xdata(end_slider.val)
            dummy = ax.fill_between(RamanShift, y1, raw.get_ydata(),
                                    where=((RamanShift >= end_slider.val) & (RamanShift <= start_slider.val)), alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])

        fig.canvas.draw_idle()

    # Connect the update function to the slider events
    start_slider.on_changed(update)
    end_slider.on_changed(update)


    # Add buttons to reset, apply, and save the baseline values
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    ResetButton = Button(resetax, 'Reset', hovercolor='0.975')
    tryax = fig.add_axes([0.69, 0.025, 0.1, 0.04])
    TryButton = Button(tryax, 'Try', hovercolor='0.975')
    doneax = fig.add_axes([0.58, 0.025, 0.1, 0.04])
    DoneButton = Button(doneax, 'Done', hovercolor='0.975')

    # Define functions for button callbacks
    def reset(event):
        dummy = ax.fill_between(RamanShift, y1, SpectralData[:, 0],
                                where=((RamanShift >= init_start) & (RamanShift <= init_end)), alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])
        start.set_xdata(init_start)
        end.set_xdata(init_end)
        raw.set_ydata(SpectralData[:,0])
        start_slider.reset()
        end_slider.reset()
        ax.set_ylim(-1, 1.1 * SpectralData[:, 0].max())
        fig.canvas.draw_idle()



    def apply_norm(event):
        positions = (np.abs(RamanShift-start_slider.val).argmin(),np.abs(RamanShift-end_slider.val).argmin())
        position = sorted(positions)
        y2 = PyMMCClass.normalise_area(SpectralData[:,0],*position)

        raw.set_ydata(y2)
        ax.set_ylim(0.95*y2.min(),1.05*y2.max())
        dummy = ax.fill_between(RamanShift, y1, raw.get_ydata(),
                                where=((RamanShift >= RamanShift[position[0]]) & (RamanShift <= RamanShift[position[1]])), alpha = 0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])
        fig.canvas.draw_idle()

    def save_vals(event):
        plt.close()
        bounds = (start_slider.val,end_slider.val)
        return sorted(bounds)

    # Connect the button callbacks to the button events
    ResetButton.on_clicked(reset)
    TryButton.on_clicked(apply_norm)
    DoneButton.on_clicked(save_vals)
    plt.show()
    bounds = (start_slider.val, end_slider.val)
    return sorted(bounds)

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
        if len(labels) == 0 and num_spectra == 1:
            labels = [self.directory[-1].replace('.sif','')]
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

                if np.all(np.logical_and(bounds >= self.RamanShift.min(), bounds <= self.RamanShift.max())):
                    start_pos, end_pos = np.abs(self.RamanShift - bounds[0]).argmin(), np.abs(
                        self.RamanShift - bounds[1]).argmin()
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

    def integrate(self, interactive = False, start, end):
        '''temporarily empty'''
    def plot_integral_kinetic(self):
        '''temporarily empty'''

    def save_changes(self, dirpath = self.directory[:-1], filename = self.directory[-1]):
        '''temporarily empty'''
        





