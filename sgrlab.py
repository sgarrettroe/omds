import numpy as np
from scipy.io import loadmat
from base import Spectrum, Response, Axis, Polarization, UNITS, OutputterHDF5

file_name = 'sgrlab/data_10P_90I.mat'
mat_dict = loadmat(file_name, simplify_cells=True)
dat_dict = mat_dict['data']

#
# extract information
#
n1, n3 = dat_dict[0]['R'].shape
n2 = len(dat_dict)
R = np.zeros(shape=(n1, n2, n3))
for ii in range(len(dat_dict)):
    R[:, ii, :] = dat_dict[ii]['R']
resp = Response(R, kind='absorptive', scale='mOD')

polarization = dat_dict[0]['polarization']
pols = []
for p in polarization:
    pols.append(Polarization(p.upper()))

w1 = dat_dict[0]['w1']
t2 = np.array([d['t2'] for d in dat_dict])
w3 = dat_dict[0]['w3']


print(R.shape)
print(w1.shape)
print(t2.shape)
print(w3.shape)
print(pols)

#
# pack it up
#
opt = {'fftshift': True}
x1 = Axis(w1, units=UNITS.WAVENUMBERS, options=opt)
x2 = Axis(t2, units=UNITS.FS)
x3 = Axis(w3, units=UNITS.WAVENUMBERS)

s = Spectrum(responses=resp, pols=pols, axes=[x1, x2, x3])

o = OutputterHDF5()
o.output(s, 'sgrlab.h5')

print('done')
