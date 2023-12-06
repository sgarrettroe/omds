"""This will be a module to demonstrate a python reference implementation of omds atomic data structures and their
output in canonical file format (HDF5).
"""
import numpy as np
import scipy.constants as constants
from pprint import pprint

TIME_DOMAIN = 'TIME'
FREQUENCY_DOMAIN = 'FREQ'


class Axis:
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


# Ideas for unit tests follow. These match my matlab code to the precision of c
t = np.arange(32, dtype=float)
options1 = {'zeropadded_length': len(t)}
dim1 = Axis(t, 'fs', options=options1)
pprint(dim1.x)
dim1.t_to_w()
pprint(dim1.x)
# array([    0.        ,  1042.38779749,  2084.77559499,  3127.16339248,
#        4169.55118998,  5211.93898747,  6254.32678497,  7296.71458246,
#        8339.10237995,  9381.49017745, 10423.87797494, 11466.26577244,
#       12508.65356993, 13551.04136742, 14593.42916492, 15635.81696241,
#       16678.20475991, 17720.5925574 , 18762.9803549 , 19805.36815239,
#       20847.75594988, 21890.14374738, 22932.53154487, 23974.91934237,
#       25017.30713986, 26059.69493736, 27102.08273485, 28144.47053234,
#       29186.85832984, 30229.24612733, 31271.63392483, 32314.02172232])

options2 = {}
dim1 = Axis(t, 'fs', options=options2)
pprint(dim1.x)
dim1.t_to_w()
pprint(dim1.x)

options3 = {'fftshift': True}
dim1 = Axis(t, 'fs', options=options3)
pprint(dim1.x)
dim1.t_to_w()
pprint(dim1.x)

options4 = {'fftshift': True, 'zeropadded_length': 33}
t = np.arange(33, dtype=float)
dim1 = Axis(t, 'fs', options=options4)
pprint(dim1.x)
dim1.t_to_w()
pprint(dim1.x)
