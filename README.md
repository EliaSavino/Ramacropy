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
This documentation will not only show you how to properly use this package but also how to handle Raman data as you should.

### Importing the package
Make sure you have the Ramacrompy folder in your working directory and add 
``` python
from Ramacrompy.Ramacrompy import Spectra
```
do not go change the code in there. if you do, stuff will not work anymore. just use it in your script.


### Loading data
You have some data in a sif file from the lab, put it in your working directory (or in a subdirectory). You can load it in python with:

```python
my_spec = Spectra('Path/to/file.sif', laser_wavelength = 785.0)
```
now, this script is able to read, sif files from andor spectrometers. .pkl and .csv files generated from THIS SCRIPT only (it could change if we realise that the files we get from the lab will be csv). 
The laser_wavelength parameter is defaulted at 785 nm (as that's what well use in the lab) change it only if we used a different laser.

### Exploratory Plotting
Once the data is loaded, you should plot it to see what you are working with. you can do this in two ways, depending if your data is a kinetic series or a single specrum.

for kinetic series use:
```python
my_spec.plot_kinetic()
```

you will get the follwing:
![Alt Text](\ReadmeIMG\Plot_rawKin.png)

for single spectra use: 

```python
my_spec.plot_few()
```

which yields: 
![Alt Text](\ReadmeIMG\Plot_rawSin.png)





