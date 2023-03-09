
from Ramacrompy import Spectra


Spec = Spectra('./DataFiles/20230301/TRC_434_M4_before.sif')
Spec2 = Spectra('./DataFiles/20230301/TRC_434_M4_after.sif')
Spec.integrate(start=650.0, end = 800.0)
Spec2.integrate(start=1000.0, end = 1030.0)

Spec.plot_integral_single(other_spectra=[Spec2], labels=['jeofrry'])

