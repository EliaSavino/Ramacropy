#import package
from Ramacropy.Ramacropy import Spectra
# load data
Spec1 = Spectra('DataFiles/20230301/Example_Raw_Before.sif')
Spec2 = Spectra('DataFiles/20230301/Example_Raw_After.sif')
# baselien correct
Spec1.baseline(coarsness=0.3)
Spec2.baseline(interactive=True)
#normalise
Spec1.normalise(method = 'area', start=800,end = 850)
Spec2.normalise(method = 'area', start=800,end = 850)
# integrate
Spec1.integrate(interactive=True)
Spec2.integrate(start = 1060, end=1160)
# plot
Spec1.plot_integral_single(other_spectra=[Spec2], labels=['After','Before'])
Spec1.plot_few(other_spectra=[Spec2], labels=['After','Before'])
# save
Spec1.save_changes()
Spec2.save_changes(filename='Example_Processed_after.csv')