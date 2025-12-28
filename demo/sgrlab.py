from omds_io.inputter_sgrlab import InputterSGRLab
from omds_io.outputter_hdf5 import OutputterHDF5

file_name = '../data/sgrlab/data_10P_90I.mat'


i = InputterSGRLab()
s = i.input(file_name)

o = OutputterHDF5()
o.output(s, 'sgrlab.h5')

print('done')
