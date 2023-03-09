
from Ramacropy.Ramacropy import Spectra

Spec = Spectra('DataFiles/20230301/Example_Raw_After.sif')


# Spec.baseline(interactive=True)
Spec.normalise(method='area',interactive=True)

