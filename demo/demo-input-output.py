from base import *

# below here is testing and debugging
t = np.arange(32, dtype=float)
opts = {'fftshift': True}
dim = Axis(t, UNITS.FS, options=opts)

filename = '../tmp.h5'
# try to clean up from last time
try:
    os.remove(filename)
    logger.info(f'removing {filename}')
except FileNotFoundError:
    pass

o = OutputterHDF5()

# start a file with 3 dimension dataseries (axes), some being nested
o.output([dim, dim, dim], filename)

logger.debug(f'reading h5 file {filename}')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['x1'].attrs.items()))
    myh5disp(f)

# add another axis under raw/
o.output([dim], filename, root='raw', access_mode='a')

with h5py.File(filename, 'r') as f:
    logger.debug('h5/raw keys found: ' + pformat(f['raw'].keys()))
    logger.debug(pformat(f['raw'].attrs.items()))
    myh5disp(f)


pol = Polarization('X')
pprint(pol)
pprint(pol.pol)
o.output([pol, pol, pol, pol], filename, access_mode='a')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['pol1'].attrs.items()))
    myh5disp(f)


resp = Response(data=np.zeros((3, 3, 3)), kind='absorptive', scale='mOD')
o.output(resp, filename, access_mode='a')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['R1'].attrs.items()))
    myh5disp(f)


logger.debug(Polarization.angles_to_stokes([MAGIC_ANGLE, 0]))
pprint(Polarization.angles_to_stokes([MAGIC_ANGLE, 0])[2])

# moving toward 2D spectrum
t1 = np.arange(3, dtype=float)
t2 = np.arange(3, dtype=float)
t3 = np.arange(3, dtype=float)
opts = {'fftshift': True}
x1 = Axis(t1, UNITS.FS, options=opts)
x2 = Axis(t2, UNITS.FS, options=opts)
x3 = Axis(t3, UNITS.FS, options=opts)


# single spectrum
print('Single spectrum:\n'+'-'*8)
spec = Spectrum(responses=[resp, resp], axes=[x1, x2, x3],
                pols=[pol, pol, pol, pol])
filename = '../tmp2.h5'
try:
    os.remove(filename)
    logger.info(f'removing {filename}')
except FileNotFoundError:
    pass

o.output(spec, filename, access_mode='a', root='/spectrum')

# list of spectra
with h5py.File(filename, 'r') as f:
    myh5disp(f)

filename = '../tmp3.h5'
try:
    os.remove(filename)
    logger.info(f'removing {filename}')
except FileNotFoundError:
    pass

o.output([spec, spec], filename)
print('Multiple spectra:\n'+'-'*8)
with h5py.File(filename, 'r') as f:
    myh5disp(f)


# test groups
group1 = MyOmdsDatagroupObj([spec, spec])

filename = '../tmp4.h5'
try:
    os.remove(filename)
    logger.info(f'removing {filename}')
except FileNotFoundError:
    pass

o.output(group1, filename)
print('Single group:\n'+'-'*8)
with h5py.File(filename, 'r') as f:
    myh5disp(f)

group2 = MyOmdsDatagroupObj([group1, spec, spec])
o.output(group2, filename)
print('Nested group and spectrum:\n'+'-'*8)
with h5py.File(filename, 'r') as f:
    myh5disp(f)

# reading files is probably the next big thing...
i = InputterHDF5()
uh = i.input('tmp.h5')
pprint(uh)

print(pol)
print('done')
# --- last line
