# PyMMC
 Python package to handle MMC data for course 2022/2023

## Scope of the Package
Dear Student of MMC practical 2022/2023, as you TA, I highly encourge (read as: require you) to use Python to clean up, analyse and plot the data you have gathered for the Raman experiment of this course. However, it would be too time consuming for all of you to write your own scripts, and it would take away your focus from the core concepts of the course. After all this is a macromolecular chemistry course not an introduction to Python one. Therefore, I wrote you a neat little package that you can use to read the data in CSV files read from the experiments and perform data processing (baselining, normalisation, noise reduction and cosmic removal), data analysis (integrals and center of mass) and data plotting (single spectra, spectra over time, integral over time). This way when I read your reports what you will get judged on is your scientific, critical analysis, and writing skills (or lack thereof) and not your ability to use Excel.

You are of course allowed to not use this package and write your own, modify it or do anything else with it (open source with MIT licence). For data processing/analysis you can, if you prefer use Spectragryph on the UPV. However, screenshots of your UPV will not help your grades, so take care that the plots you present look good and make sense. 
My legal team has notified me that I technically cannot force you to use python, here's a few alternatives that are also accepted: Matlab, Javascript, Julia, C++(good luck with this one) or whitespace. Excel is also allowed(not recommended) but please take care that your plots don't look disgusting and actually provide all the information needed. 

**_For questions and debugging issues (with limitations) I'm available at e.savino@rug.nl_**

## System Reqirements
For this package no environment nor python executable is provided, You are responsible to have Python 3 on your machine and a way to manage your packages. I recommend the combination of Pip and Conda for your package and environment management. If you don't know how to do any of this, read the Python documentation, If you don't know what python is, start from the definition of a computer and work your way forward.

Required packages:
- Matplotlib
- Numpy
- Pandas
- Scipy  
- sif_parser (this you can download with pip install sif_parser), all rights reserved to https://github.com/fujiisoup

## Documentation
This documentation not only provides instructions on how to use this package effectively, but also offers guidance on handling Raman data in a best(at the best of my knowledge) practice manner.

### Importing the package
Ensure that you have the Ramacrompy folder in your working directory and add the following code to your script:
``` python
from Ramacrompy.Ramacrompy import Spectra
```
Do not make changes to the code within the folder, as doing so may cause issues with its functionality. Instead, simply import and use the package in your own script.


### Loading data
If you have data in a sif file from the lab, place it in your working directory (or in a subdirectory). You can load it into Python using the following code:
```
python
my_spec = Spectra('Path/to/file.sif', laser_wavelength=785.0)
```
This script is capable of reading sif files from Andor spectrometers, as well as .pkl and .csv files generated only from this script (though this may change if we find that the lab provides csv files). The laser_wavelength parameter defaults to 785 nm (as that's what we use in the lab), so only change it if a different laser was used.

### Exploratory Plotting
Once the data is loaded, you should plot it to see what you are working with. you can do this in two ways, depending if your data is a kinetic series or a single spectrum.

for kinetic series use:
```python
my_spec.plot_kinetic()
```

you will get the follwing:
![Alt Text](./ReadmeIMG/Plot_rawKin.png)

for single spectra use: 

```python
my_spec.plot_few()
```

which yields: 
![Alt Text](./ReadmeIMG/Plot_rawSin.png)

of course you can 

### Data Processing
In data processing, often, less is more. So the tools you have at your disposal are baseline correction, integration (by peak or by area) and spike removal. In order you should first do baseline correction, then normalisation and maybe (only if you really have a lot of spikes) spike removal. 

#### Baseline correction
The baseline correction has takes three parameters, coarsness, angle and offset. the first is an indication of how straight your baseline is, angle is what is the angle of the baseline, and offset is how far up or down do you need the baseline. If you don't know (and i sugeest you do this every time you have a new piece of data) you can run the baseline routine interactively

if you know what parameters you need:

```python
my_spec.baseline(coarsness=0.3, angle = 12, offset = 300)
```

if you don't know you can use:

```python
my_spec.baseline(interactive = True)
```
and you'll get the following interactive plot:
![Alt Text](./ReadmeIMG/Baseline_int1.png)

adjust the sliders until you get a baseline that looks satisfactory, like so:
![Alt Text](./ReadmeIMG/Baseline_int2.png)

To apply the baseline correction to your spectra, click the 'Try' button. If you want to start over, click the 'Reset' button. Once you're satisfied with the correction, click the 'Done' button to confirm.

#### Normalisation
The normalisation method is similar to the baseline, you first have to decide which method to use, either 'area' or 'peak'. The choiche is yours. Again it is possible to run this interactively if you're not sure where to put your bounds/peak. 
if you decide against the interactive method you cand do so as follows:
for area:
```python
my_spec.normalise(method='area',start=700.0,end = 750.0)
```
where start and end are the bounds of your normalisation in Raman shift (cm<sup>-1</sup>)
for peak:
```python
my_spec.normalise(method='peak',peak=700.0)
```
if you decide to run interactively, you can do, for area:
```python
my_spec.normalise(method='area',interactive=True)
```
which will open the following window:
![Alt Text](./ReadmeIMG/normalise_area.png)

use the sliders to decide your bounds of integration, the buttons do what explained previously.

for interactive peak use:
```python
my_spec.normalise(method='peak',interactive=True)
```
which will open the following:
![Alt Text](./ReadmeIMG/normalise_peak.png)

use the slider to choose your peak position, and use the buttons to apply, reset or confirm

#### Spike removal
It can happen that the sensor in the spectrometer picks up random cosmic rays passing through it, this show up in the spectra as "cosmics", or spikes, you can recognise them as they are very thin, only last 1 spectrum, and change without any recognisable pattern.
To attempt at removing them you can use, 
```python
my_spec.spike_removal()
```

if it doesn't remove them all, it's fine. just ignore them, (acknowledge this in your analysis)

### Data saving
Congrats, you have gotten though your data processing, before we discuss how to do data analysis, let's discuss how to save your data. 
you can save the data as a .pkl file, which is a binary file that saves all the information in your data (raw data, processed data, time stamps, spectrometer conditions etc etc) but is not human readable (this is the default) or in a human readable CSV. To do so use:
```python
my_spec.save_changes(dirpath = 'Where/Save', filename = 'NameFile.pkl(or.csv)')
```
dirpath and filename arguments are optional, if the dirpath is missing it will save in the directory where the original file was, and if the filename is missing it will save a .pkl file with the same name as the original file.

BE CAREFUL, THIS WILL OVERWRITE FILES WITHOUT WARNING YOU!!! (you have been warned)

the files generated with this command are can be opened with this program (check above)

### Data analysis
You are provided only one method of data analysis (as i think that for the scope of this course is more than enough), which is integration, you of course are also provided (i hope) your common sense, which is paramount in you understanding what is going on in your spectra!

#### Integration
Integration measures the area of your spectrum within certain bounds. The code works exactly as the normalisation by area 