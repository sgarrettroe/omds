"""This will be a module to demonstrate a python reference
implementation of omds atomic data structures and their output in
canonical file format (HDF5).
"""
import numpy as np
import scipy.constants as constants
import h5py
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from enum import StrEnum, auto
import logging
from pprint import pformat
from typing import Tuple

# for debugging / testing (can/should be removed)
import os
from pprint import pprint

# set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s'
)
logger = logging.getLogger('omds.atomic_data_draft')
logger.setLevel(logging.DEBUG)


class Domain(StrEnum):
    """Indicator of time domain or frequency domain data.

    An StrEnum can be tested with `is` or `==`. """
    TIME = auto()
    FREQUENCY = auto()


class MyOmdsDatasetObj:
    """Base class that objects that output data should subclass.

    The interface is `obj.dataset` which should return a dict of
    {'data':val,'dtype':val,'attr':dict}. Subclasses must define
    _get_dataset which must return that dict structure or list of them.

    Properties
    ----------
    dataset: dict
        Dictionary that describes the dataset in the form
        ``d = {'basename': <name>,
               'data': <data scalar or array>,
               'dtype': <data type string>,
               'attr': <dict of data attributes>}``

    Methods
    -------
    _get_dataset: dict
        Return the dataset dictionary. Must be overridden by subclass.
    """
    @property
    def dataset(self):
        return self._get_dataset()

    def _get_dataset(self) -> dict:
        """Function that should be created to return a dataset
        dictionary or list of them."""
        raise NotImplementedError("Please Implement this method")


PolarizationTuple = namedtuple('Polarization',
                               ['name',
                                'Jones3',
                                'cartesian',
                                'angles',
                                'Stokes'])
POLARIZATION_TYPE = np.dtype([('name', h5py.string_dtype()),
                              ('Jones3', 'c', (3,)),
                              ('cartesian', 'f', (3,)),
                              ('angles', 'f', (2,)),
                              ('Stokes', 'f', (4,)),
                              ])
MAGIC_ANGLE = np.arccos(np.sqrt(1/3))


class Polarization(MyOmdsDatasetObj):
    """
    Collect information about the polarization of light.

    Example
    -------
    Polarization XXXX could be indicated with

    .. code-block:: python

        pol = [Polarization('X'), Polarization.('X'),
               Polarization.('X'), Polarization.('X')]

    and XXYY with

    .. code-block:: python

        pol = [Polarization('X'), Polarization('X'),
               Polarization('Y'), Polarization('Y')]

    additional polarizations can be added by
    .. code-block:: python

        setattr(Polarization, <attribute_name>,
            <wrapped_polarization_tuple>)

    for example, +45° linear polarization travelling in the Z-direction
    could be coded
    .. code-block:: python

        pt = np.array([PolarizationTuple(name='lin_45_Z',
                          Jones3=np.array([1, 1, 0])/np.sqrt(2),
                          cartesian=[0, 0, 1],
                          angles=[np.pi/4, 0],
                          Stokes=[1, 0, 1, 0])],
                      dtype=POLARIZATION_TYPE)
        setattr(Polarization,'lin_45_Z',pt)

    Properties
    ----------
    X : np.ndarray
        Linearly polarized X travelling in Z-direction
    Y : np.ndarray
        Linearly polarized Y travelling in Z-direction
    Z : np.ndarray
        Linearly polarized travelling in X-direction
    R : np.ndarray
        Right circularly polarized light travelling in the Z-direction
    L : np.ndarray
        Left circularly polarized light travelling in the Z-direction
    M : np.ndarray
        Light polarized at the magic angle relative to the X-axis and
        travelling in the Z-direction
    U : np.ndarray
        Unpolarized light travelling in the Z-direction
    R_X : np.ndarray
        Right circularly polarized light travelling in the X-direction
    L_X : np.ndarray
        Left circularly polarized light travelling in the X-direction
    R_Y : np.ndarray
        Right circularly polarized light travelling in the Y-direction
    L_Y : np.ndarray
        Left circularly polarized light travelling in the Y-direction
    R_Z : np.ndarray
        Right circularly polarized light travelling in the Z-direction
    L_Z : np.ndarray
        Left circularly polarized light travelling in the Z-direction

    Methods
    -------
    _get_dataset: dict
        Return a dictionary to save the data

    Notes
    -----
    The default direction of travel for the light is the Z-axis, used
    in the single letter codes 'X', 'Y', 'L', 'R'.

    References
    ----------
    See DOI: 10.1103/PhysRevA.90.023809 for a full 3D treatment.
    """
    X = np.array([PolarizationTuple(name='X',
                                    Jones3=np.array([1, 0, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, 0]),
                                    Stokes=np.array([1, 1, 0, 0]))],
                 dtype=POLARIZATION_TYPE)
    Y = np.array([PolarizationTuple(name='Y',
                                    Jones3=np.array([0, 1, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([np.pi/2, 0]),
                                    Stokes=np.array([1, -1, 0, 0]))],
                 dtype=POLARIZATION_TYPE)
    Z = np.array([PolarizationTuple(name='Z',
                                    Jones3=np.array([0, 0, 1]),
                                    cartesian=np.array([1, 0, 0]),
                                    angles=np.array([0, 0]),
                                    Stokes=np.array([1, 1, 0, 0]))],
                 dtype=POLARIZATION_TYPE)
    M = np.array(
        [PolarizationTuple(name='Z',
                                Jones3=np.array([np.cos(MAGIC_ANGLE),
                                                 np.sin(MAGIC_ANGLE), 0]),
                                cartesian=np.array([0, 0, 1]),
                                angles=np.array([MAGIC_ANGLE, 0]),
                                Stokes=np.array([1,
                                                 -1/3,
                                                 0.9428090415820634,
                                                 0]))],
        dtype=POLARIZATION_TYPE)
    U = np.array([PolarizationTuple(name='U',
                                    Jones3=None,
                                    cartesian=np.array([0, 0, 1]),
                                    angles=None,
                                    Stokes=np.array([1, 0, 0, 0]))],
                 dtype=POLARIZATION_TYPE)
    R = np.array([PolarizationTuple(name='R',
                                    Jones3=np.array([1, 1j, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, -np.pi/4]),
                                    Stokes=np.array([1, 0, 0, 1]))],
                 dtype=POLARIZATION_TYPE)
    L = np.array([PolarizationTuple(name='R',
                                    Jones3=np.array([1, 1j, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, np.pi/4]),
                                    Stokes=np.array([1, 0, 0, -1]))],
                 dtype=POLARIZATION_TYPE)
    R_X = np.array([PolarizationTuple(name='R_X',
                                      Jones3=np.array([0, 1, -1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE)
    L_X = np.array([PolarizationTuple(name='L_X',
                                      Jones3=np.array([0, 1, 1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE)
    R_Y = np.array([PolarizationTuple(name='R_Y',
                                      Jones3=np.array([1, 0, -1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE)
    L_Y = np.array([PolarizationTuple(name='L_Y',
                                      Jones3=np.array([1, 0, 1j]) / np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE)
    R_Z = np.array([PolarizationTuple(name='R_Z',
                                      Jones3=np.array([1, -1j, 0])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE)
    L_Z = np.array([PolarizationTuple(name='L_Z',
                                      Jones3=np.array([1, 1j, 0])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE)

    def __init__(self, pol_in):
        self.pol = getattr(self, pol_in)

    @staticmethod
    def jones_to_stokes(jones) -> np.ndarray:
        """Convert 2D Jones vector to Stokes vector.
        """
        z = np.array(jones)
        s0 = 1
        s1 = np.abs(z[0])**2 - np.abs(z[1])**2
        s2 = 2*np.real(z[0]*z[1].conj())
        s3 = 2*np.imag(z[0]*z[1].conj())
        return np.array([s0, s1, s2, s3])

    @staticmethod
    def angles_to_stokes(angles, intensity=1, p=1) -> np.ndarray:
        """Convert polarization ellipse parameters to Stokes vector.
        """
        psi = angles[0]
        chi = angles[1]
        s = np.array([
            intensity,
            intensity*p*np.cos(2*psi)*np.cos(2*chi),
            intensity*p*np.sin(2*psi)*np.cos(2*chi),
            -intensity*p*np.sin(2*chi)
        ])
        return s

    @staticmethod
    def stokes_to_jones(S) -> Tuple[np.ndarray, float]:
        # Calculate the degree of polarization
        p = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2) / S[1]

        I = 1
        Q = S[1] / (S[0] * p)
        U = S[2] / (S[0] * p)
        V = S[3] / (S[0] * p)

        a = np.sqrt((1 + Q) / 2)
        if a == 0:
            b = 1
        else:
            b = U / (2 * a) - 1j * V / (2 * a)
        j = np.sqrt(I * p) * np.array([a, b])
        return j, p

    def _get_dataset(self) -> dict:
        return {'basename': 'pol',
                'data': self.pol,
                'dtype': POLARIZATION_TYPE,
                'attr': {'label': self.pol[0]}}


class Axis(MyOmdsDatasetObj):
    """
    A time or frequency axis object.
    """
    def __init__(self, x: np.ndarray,
                 units: str,
                 current_domain: Domain = Domain.TIME,
                 options=None):
        self.x = x
        self.units = units
        self.current_domain = current_domain
        self.time: np.ndarray  # ToDo: decide to keep both time and freq internally? or recalculate?
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

        if current_domain is Domain.TIME:
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
        if self.current_domain is Domain.TIME:
            self.t_to_w()

        if self.current_domain is Domain.FREQUENCY:
            self.w_to_t()

    def unit_conversion(self, new_units):
        # change units and calculate the unit conversion on the data
        pass

    def load_known_units(self, uri_list=None):
        """ Load units and conversion factors from QUDT. But I don't
        know how to do that yet! For now use scipy.constants.
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
        d = {'basename': 'x',
             'data': self.x,
             'dtype': self.x.dtype,
             'attr': {'units': self.units} | self.options}
        return d


class Spectrum(MyOmdsDatasetObj):
    def __init__(self):
        self.data = np.zeros((3, 3, 3))

    def _get_dataset(self) -> dict:
        return {'basename': 'R',
                'data': self.data,
                'dtype': self.data.dtype,
                'attr': {'units': 'mOD',
                         }
                }


class Outputter:
    """Base class for output functionality. All output methods should
    inherit from this class.
    """
    # take response function in and provide fxn to write output file

    def output(self, obj, filename, root='/'):
        pass


class OutputterHDF5(Outputter):
    """Class to write output to an HDF5 file.
    """
    # default output mechanism, writes HDF5 file
    def output(self, obj_in, filename, root='/', access_mode='w') -> None:
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

        # dictionary of how many times each basename, which is used to
        # label datasets uniquely. returns an integer 0 for new keys.
        name_counts = defaultdict(int)

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


# below here is testing and debugging
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

pol = Polarization('X')
pprint(pol)
pprint(pol.pol)
o.output([pol, pol], filename, access_mode='a')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['pol1'].attrs.items()))

s = Spectrum()
o.output(s, filename, access_mode='a')
with h5py.File(filename, 'r') as f:
    logger.debug('h5 keys found: ' + pformat(f.keys()))
    logger.debug(pformat(f['R1'].attrs.items()))

logger.debug(Polarization.angles_to_stokes([MAGIC_ANGLE, 0]))
pprint(Polarization.angles_to_stokes([MAGIC_ANGLE, 0])[2])
# --- last line
