"""This will be a module to demonstrate a python reference implementation of omds atomic data structures and their
output in canonical file format (HDF5).
"""
import numpy as np
import scipy.constants as constants
import h5py
from collections import defaultdict
from collections.abc import Iterable
import logging

# for debugging / testing
from pprint import pformat
import os

TIME_DOMAIN = 'TIME'
FREQUENCY_DOMAIN = 'FREQ'

# set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s')
logger = logging.getLogger('omds.atomic_data_draft')
logger.setLevel(logging.DEBUG)


class MyOmdsDatasetObj:
    """Base class that objects that output data should subclass.

    The interface is obj.dataset which should return a dict of {'data':val,'dtype':val,'attr':dict}.
    Subclasses must define _get_dataset which must return that dict structure or list of them.

    dataset = {'basename':<name>,
               'data':<data scalar or array>,
               'dtype':<data type string>,
               'attr':<dict of data attributes>}
    """
    @property
    def dataset(self):
        return self._get_dataset()

    def _get_dataset(self) -> dict:
        """Function that should be created to return a dataset dictionary or list of them."""
        raise NotImplementedError("Please Implement this method")


class Axis(MyOmdsDatasetObj):
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
        # ToDo: figure out how to scrape URIs using rdflib
        # ToDo: add units eV, atomic units, what else?
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

    def _get_dataset(self) -> dict:
        d = {'basename':'x',
             'data': self.x,
             'dtype': 'f',
             'attr': {'units': self.units} | self.options}
        return d


class Spectrum(MyOmdsDatasetObj):
    def _get_dataset(self) -> dict:
        return {'data': 44.0,
                'dtype': 'f',
                'attr': {'units': 'G-PER-MOL'}}


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
    def output(self, obj_in, filename, root='/', access_mode = 'w')->None:
        """Output HDF5 file that saves the dataset or sets of them.

        Parameters
        ----------
        obj_in : MyOmdsDatasetObj or list of MyOmdsDatasetObj
            The input objects to be processed. They must have an
            attribute dataset. Dataset must be a dictionary with keys
            "basename", "data", and "dtype".
        filename : str
            The name of the output file.
        root :
            The base group of the dataset(s).
        access_mode : {'w','a'}, optional
            The mode used to open the output file. The default is 'w',
            which overwrites and existing file.

        Returns
        -------
        None
        """

        # make sure there is one trailing / in the root name
        root = root.rstrip('/') + '/'

        # dictionary of how many times each basename, which is used to label datasets uniquely
        name_counts = defaultdict(int)  # returns an empty integer for new key items

        def process_item(obj):
            if isinstance(obj, Iterable):
                for this_obj in obj:
                    process_item(this_obj)
            else:
                dset = obj.dataset
                root_basename = root.lstrip('/') + dset["basename"]
                name_counts[root_basename] += 1
                full_name = f'{root_basename}{name_counts[root_basename]}'
                h5dset = f.create_dataset(full_name,
                                          dset['data'].shape,
                                          dtype=dset['dtype'],
                                          data=dset['data'])
                for (key, val) in dset['attr'].items():
                    h5dset.attrs[key] = val

        with h5py.File(filename, access_mode) as f:
            process_item(obj_in)  # recursively process input (depth first)


t = np.arange(32, dtype=float)
opts = {'fftshift': True}
dim = Axis(t, 'fs', options=opts)

filename = 'tmp.h5'
# try to clean up from last time
try:
    os.remove(filename)
    logger.info(f'removing {filename}')
except FileNotFoundError:
    pass

o = OutputterHDF5()

# start a file with 3 dimension datasets (axes), some being nested
o.output([dim, [dim, dim]], filename)

logger.debug(f'reading h5 file {filename}')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['x1'].attrs.items()))

# add another axis under raw/
o.output([dim], filename, root='raw', access_mode='a')

with h5py.File(filename, 'r') as f:
    logger.debug('h5/raw keys found: ' + pformat(f['raw'].keys()))
    logger.debug(pformat(f['raw'].attrs.items()))

