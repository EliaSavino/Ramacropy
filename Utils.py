import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button
from scipy.signal import savgol_filter
import numpy as np
import uuid
import configparser
import os


def GenID():
    '''Don't worry about this function, you don't need it'''
    config_file = 'config.ini'

    # Check if configuration file exists, otherwise create it
    if not os.path.exists(config_file):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'user_id': str(uuid.uuid4())}
        with open(config_file, 'w') as f:
            config.write(f)

    # Read the user ID from the configuration file
    config = configparser.ConfigParser()
    config.read(config_file)
    user_id = config['DEFAULT']['user_id']
    return user_id

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
    calcax = fig.add_axes([0.69, 0.025, 0.1, 0.04])
    CalcButton = Button(calcax, 'Calculate', hovercolor='0.975')

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


    def save_vals(event):
        plt.close()
        bounds = (start_slider.val,end_slider.val)
        return sorted(bounds)

    # Connect the button callbacks to the button events
    ResetButton.on_clicked(reset)
    CalcButton.on_clicked(save_vals)
    plt.show()
    bounds = (start_slider.val, end_slider.val)
    return sorted(bounds)
