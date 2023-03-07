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

Spec = PyMMCClass.Spectra('./DataFiles/20230301/TRC_434_M4_after.sif')



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
        dummy = ax.fill_between(RamanShift, y1, y2,
                                where=((RamanShift >= RamanShift[position[0]]) & (RamanShift <= RamanShift[position[1]])), alpha = 0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        fill.set_paths([dp.vertices])
        raw.set_ydata(y2)
        ax.set_ylim(0.95*y2.min(),1.05*y2.max())
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


peak = InteractiveNormaliseArea(Spec.RamanShift,Spec.SpectralData)