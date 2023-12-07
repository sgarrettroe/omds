"""This will be a module to demonstrate a python reference implementation of omds atomic data structures and their
output in canonical file format (HDF5).
"""
import numpy as np
import scipy.constants as constants
import h5py
from pprint import pprint

TIME_DOMAIN = 'TIME'
FREQUENCY_DOMAIN = 'FREQ'


class Axis():
    """
    A time or frequency axis object.
    """
    def __init__(self, x: np.ndarray, units: str, current_domain=TIME_DOMAIN, options=None):
        self.x = x
        self.units = units
        self.current_domain = current_domain
        self.time: np.ndarray  # keep both time and freq internally? or recalculate?
        self.time_units: str
        self.frequency: np.ndarray
        self.frequency_units: str
        self.known_unit_table = {}
        self.options = {
            'time_units': 'fs',
            'frequency_units': 'cm-1',
            'zeropadded_length': 2*len(self.x),
            'n_undersampling': 0,
            'fftshift': False,
        }
        if options is None:
            pass
        else:
            self.options |= options

        self.load_known_units(None)

        if current_domain == TIME_DOMAIN:
            self.time = self.x
            self.time_units = self.units
            self.frequency_units = self.options['frequency_units']
        else:
            self.frequency = self.x
            self.frequency_units = self.units
            self.time_units = self.options['time_units']

    def t_to_w(self):
        """
        Convert time axis to frequency axis.


        Returns: None

        """
        tu = self.known_unit_table['time_units']
        fu = self.known_unit_table['frequency_units']
        conversion = tu[self.units] * fu[self.frequency_units]

        n_t = self.options['zeropadded_length']
        dt = self.x[1] - self.x[0]
        fs = 1 / (dt * conversion)  # sampling frequency
        dw = 1 / (dt * conversion * n_t)
        w = np.arange(n_t) * dw
        if self.options['fftshift']:
            ind = int(np.ceil(n_t/2))
            left = w[0:ind]
            right = w[ind:] - fs
            w = np.concatenate((right, left), 0)
        self.x = w
        self.units = self.frequency_units
        return

    def w_to_t(self):
        pass

    def fft_axis(self):
        if self.current_domain == TIME_DOMAIN:
            self.t_to_w()

        if self.current_domain == FREQUENCY_DOMAIN:
            self.w_to_t()

    def unit_conversion(self, new_units):
        # change units and calculate the unit conversion on the data
        pass

    def load_known_units(self, uri_list=None):
        """ Load units and conversion factors from QUDT. But I don't know how to do that yet! For now use
        scipy.constants.
        """
        if uri_list is None:
            self.known_unit_table = {
                'time_units': {
                    'fs': constants.femto,
                    'ps': constants.pico,
                    's': 1,
                        },
                'frequency_units': {
                    'Hz': 1,
                    'THz': constants.tera,
                    'rad_per_sec': 2*constants.pi,
                    'cm-1': constants.c/constants.centi,
                    'wavenumbers': constants.c/constants.centi,
                }
            }

    def output(self):
        # print axis and label in some nice way
        pass

    def __getattr__(self, item):
        match item:
            case 'dset' | 'dataset':
                d = {'data':self.x,
                     'dtype':'f',
                     'attr':{'units':self.units} | self.options}
                return d


class Outputter:
    """Base class for output functionality. All methods should inherit from this class.
    """
    # take response function in and provide fxn to write output file

    def output(self, obj, filename, root='/'):
        pass



class OutputterHDF5(Outputter):
    """Class to write output to an HDF5 file.
    """
    # default output mechanism, writes HDF5 file
    def output(self, obj, filename, root='/'):
        match obj:
            case Axis():  # ToDo: change to look for dset ???
                with h5py.File(filename,'a') as f:
                    dset = obj.dset
                    #h5dset = f.create_dataset('x1', (len(obj.x),), dtype='f', data=obj.x)
                    h5dset = f.create_dataset(f'{root}x1', (len(dset['data']),), dtype=dset['dtype'], data=dset['data'])
                    for (key, val) in dset['attr'].items():
                        h5dset.attrs[key] = val


t = np.arange(32, dtype=float)
opts = {'fftshift': True}
dim = Axis(t, 'fs', options=opts)

o = OutputterHDF5()
o.output(dim,'tmp.h5')

with h5py.File('tmp.h5','r') as f:
    for key in f.keys():
        pprint(key)
    pprint(f['x1'].attrs())

