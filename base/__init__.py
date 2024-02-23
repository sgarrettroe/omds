"""This will be a module to demonstrate a python reference
implementation of omds atomic data structures and their output in
canonical file format (HDF5).
"""
import numpy as np
import scipy.constants as constants
import h5py
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum
import logging
from typing import Tuple
import os  # path used in manipulating HDF5 names

# for debugging / testing (can/should be removed)
from pprint import pprint

# set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s'
)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

OMDS = {'kind': {'ABSORPTIVE': 'omds:Absorptive',
                 'DISPERSIVE': 'omds:Dispersive',
                 'REPHASING': 'omds:Rephasing',
                 'NONREPHASING': 'omds:Nonrephasing',
                 'ABS': 'omds:AbsoluteValue',
                 'MIXED': 'omds:MixedKind',
                 },
        'scale': {'mOD': 'omds:ScaleMilliOD',
                  'uOD': 'omds:ScaleMicroOD',
                  'OD': 'omds:ScaleOD',
                  '%T': 'omds:ScalePercentTransmission',
                  'OD/sqrt(Hz)': 'omds:ScaleOD-SQRT-SEC',
                  # https://doi.org/10.1021/acs.analchem.2c04287
                  'OD/sqrt(cm-1)': 'omds:ScaleOD-SQRT-CentiM',
                  }
        }


# noinspection PyPep8Naming
class UNITS(Enum):
    """ Load units and conversion factors from QUDT.

    But I don't know how to do that yet! For now use scipy.constants.
    """
    # ToDo: incorporate .units functionality here
    # Time
    AS = constants.atto  # note clash of SI "as" with keyword "as"
    ATTOSECONDS = constants.atto
    FS = constants.femto
    PS = constants.pico
    NS = constants.nano
    US = constants.micro
    MS = constants.milli
    S = 1
    # Frequency
    HZ = 1
    THZ = constants.tera
    RAD_PER_SEC = 2 * constants.pi
    CM_1 = constants.c / constants.centi
    INV_CM = constants.c / constants.centi
    PER_CM = constants.c / constants.centi
    WAVENUMBERS = constants.c / constants.centi


class UNIT_TYPE(set, Enum):
    TIME_UNITS = {UNITS.ATTOSECONDS,
                  UNITS.FS,
                  UNITS.PS,
                  UNITS.NS,
                  UNITS.US,
                  UNITS.MS,
                  UNITS.S}
    FREQUENCY_UNITS = {UNITS.HZ,
                       UNITS.THZ,
                       UNITS.RAD_PER_SEC,
                       UNITS.INV_CM}


class MyOmdsDataseriesObj:
    """Base class that objects that output data should subclass.

    The interface is `obj.dataseries` which should return a dict of
    {'data':val,'dtype':val,'attr':dict}. Subclasses must define
    the dataseries getter, which must return that dict structure or
    list of them.

    Properties
    ----------
    dataseries: dict
        Dictionary that describes the dataseries in the form
        ``d = {'basename': <name>,
               'data': <data scalar or array>,
               'dtype': <data type string>,
               'attr': <dict of data attributes>}``
        Must be overridden by subclass.
    """
    basename = ''

    @property
    def dataseries(self):
        """Function that should be created to return a dataseries
        dictionary or list of them."""
        raise NotImplementedError("Please Implement this method")


class MyOmdsDatagroupObj:
    """Class for data groups.

    Data groups are containers that can contain dataseries or other
    datagroup(s).

    We're using duck typing to make this class iterable.

    Each subclass should implement a datagroup as a list (or other
    iterable) that will be used to iterate over.

    Properties
    ----------
    datagroup: list
        The data group contents.
    attributes: dict
        The attributes of the datagroup.
            ``d = {'class': str}

    """

    basename = 'Datagroup'

    def __init__(self, data_group: list = None, attributes: dict = None):
        if data_group is None:
            data_group = []
        self.datagroup = data_group
        if attributes is None:
            attributes = {}
        self._attributes = attributes

    def __iter__(self):
        return self.datagroup.__iter__()

    def __getitem__(self, item):
        return self.datagroup[item]

    @property
    def attributes(self):
        return {'class': self.__class__.__name__,
                } | self._attributes


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


class Polarization(MyOmdsDataseriesObj):
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

    Properties
    -------
    dataseries: dict
        Return a dictionary to save the data

    Notes
    -----
    The default direction of travel for the light is the Z-axis, used
    in the single letter codes 'X', 'Y', 'L', 'R'.

    References
    ----------
    See DOI: 10.1103/PhysRevA.90.023809 for a full 3D treatment.
    """
    basename = 'pol'
    polarization_dict = {
        'X': np.array([PolarizationTuple(name='X',
                                    Jones3=np.array([1, 0, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, 0]),
                                    Stokes=np.array([1, 1, 0, 0]))],
                 dtype=POLARIZATION_TYPE),
        'Y': np.array([PolarizationTuple(name='Y',
                                    Jones3=np.array([0, 1, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([np.pi/2, 0]),
                                    Stokes=np.array([1, -1, 0, 0]))],
                 dtype=POLARIZATION_TYPE),
        'Z': np.array([PolarizationTuple(name='Z',
                                    Jones3=np.array([0, 0, 1]),
                                    cartesian=np.array([1, 0, 0]),
                                    angles=np.array([0, 0]),
                                    Stokes=np.array([1, 1, 0, 0]))],
                 dtype=POLARIZATION_TYPE),
        'M': np.array(
        [PolarizationTuple(name='Z',
                                Jones3=np.array([np.cos(MAGIC_ANGLE),
                                                 np.sin(MAGIC_ANGLE), 0]),
                                cartesian=np.array([0, 0, 1]),
                                angles=np.array([MAGIC_ANGLE, 0]),
                                Stokes=np.array([1,
                                                 -1/3,
                                                 0.9428090415820634,
                                                 0]))],
        dtype=POLARIZATION_TYPE),
        'U': np.array([PolarizationTuple(name='U',
                                    Jones3=None,
                                    cartesian=np.array([0, 0, 1]),
                                    angles=None,
                                    Stokes=np.array([1, 0, 0, 0]))],
                 dtype=POLARIZATION_TYPE),
        'R': np.array([PolarizationTuple(name='R',
                                    Jones3=np.array([1, 1j, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, -np.pi/4]),
                                    Stokes=np.array([1, 0, 0, 1]))],
                 dtype=POLARIZATION_TYPE),
        'L': np.array([PolarizationTuple(name='R',
                                    Jones3=np.array([1, 1j, 0]),
                                    cartesian=np.array([0, 0, 1]),
                                    angles=np.array([0, np.pi/4]),
                                    Stokes=np.array([1, 0, 0, -1]))],
                 dtype=POLARIZATION_TYPE),
        'R_X': np.array([PolarizationTuple(name='R_X',
                                      Jones3=np.array([0, 1, -1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE),
        'L_X': np.array([PolarizationTuple(name='L_X',
                                      Jones3=np.array([0, 1, 1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE),
        'R_Y': np.array([PolarizationTuple(name='R_Y',
                                      Jones3=np.array([1, 0, -1j])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE),
        'L_Y': np.array([PolarizationTuple(name='L_Y',
                                      Jones3=np.array([1, 0, 1j]) / np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE),
        'R_Z': np.array([PolarizationTuple(name='R_Z',
                                      Jones3=np.array([1, -1j, 0])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, -np.pi/4]),
                                      Stokes=np.array([1, 0, 0, 1]))],
                   dtype=POLARIZATION_TYPE),
        'L_Z': np.array([PolarizationTuple(name='L_Z',
                                      Jones3=np.array([1, 1j, 0])/np.sqrt(2),
                                      cartesian=np.array([1, 0, 0]),
                                      angles=np.array([0, np.pi/4]),
                                      Stokes=np.array([1, 0, 0, -1]))],
                   dtype=POLARIZATION_TYPE),
    }

    def __init__(self, pol_in):
        self.pol = self.polarization_dict[pol_in]

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
    def stokes_to_jones(s) -> Tuple[np.ndarray, float]:
        # Calculate the degree of polarization
        p = np.sqrt(s[1]**2 + s[2]**2 + s[3]**2) / s[1]

        I = 1
        Q = s[1] / (s[0] * p)
        U = s[2] / (s[0] * p)
        V = s[3] / (s[0] * p)

        a = np.sqrt((1 + Q) / 2)
        if a == 0:
            b = 1
        else:
            b = U / (2 * a) - 1j * V / (2 * a)
        j = np.sqrt(I * p) * np.array([a, b])
        return j, p

    @property
    def label(self) -> str:
        return self.pol[0][0]

    @property
    def dataseries(self) -> dict:
        return {'basename': self.basename,
                'data': self.pol,
                'dtype': POLARIZATION_TYPE,
                'attr': {
                    'class': self.__class__.__name__,
                    'label': self.label},
                }

    def __str__(self):
        return f'Polarization("{self.label}")'

    def __repr__(self):
        return f'Polarization("{self.label}")'

class Axis(MyOmdsDataseriesObj):
    """
    A time or frequency axis object.
    """
    basename = 'x'

    def __init__(self, x: np.ndarray,
                 units,
                 defaults=None,
                 options=None):
        self.x = x
        self.units = units
        self.defaults = {
            'time_units': UNITS.FS,
            'frequency_units': UNITS.INV_CM,
        }
        self.options = {
            'zeropadded_length': 2*len(self.x),
            'n_undersampling': 0,
            'fftshift': False,
            'rotating_frame': 0,
        }

        if defaults is None:
            pass
        else:
            self.defaults |= defaults

        if options is None:
            pass
        else:
            self.options |= options

    def __str__(self):
        return f'Axis(...) shape:{self.x.shape} units:{self.units}'

    def __repr__(self):
        return f'Axis(...) shape:{self.x.shape} units:{self.units}'

    def t_to_w(self, frequency_units=None):
        """
        Convert time axis to frequency axis.


        Returns: None

        """
        tu = self.units
        if frequency_units is None:
            fu = self.defaults['frequency_units']
        else:
            fu = frequency_units

        conversion = tu.value * fu.value

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
        self.units = frequency_units
        return

    def w_to_t(self, time_units=None):
        fu = self.units
        if time_units is None:
            tu = self.defaults['time_units']
        else:
            tu = time_units

        conversion = tu.value * fu.value

        logger.error('Frequency to time not yet implemented')
        raise NotImplementedError

    def fft_axis(self):
        if self.units in UNIT_TYPE.TIME_UNITS:
            self.t_to_w()

        if self.units in UNIT_TYPE.FREQUENCY_UNITS:
            self.w_to_t()

    def unit_conversion(self, new_units):
        # change units and calculate the unit conversion on the data
        raise NotImplementedError

    def output(self):
        # print axis and label in some nice way
        pass

    @property
    def dataseries(self) -> dict:
        d = {'basename': self.basename,
             'data': self.x,
             'dtype': self.x.dtype,
             # the | character below merges dicts
             'attr': {'class': self.__class__.__name__,
                      'units': self.units.name} | self.options,
             }
        return d


class Response(MyOmdsDataseriesObj):
    basename = 'R'

    def __init__(self, data: np.ndarray, kind: str, scale: str = 'mOD'):
        self.data = data
        self.kind = OMDS['kind'][kind.lstrip('omds:').upper()]
        self.scale = OMDS['scale'][scale.lstrip('omds:Scale')]

    def __str__(self):
        return f'Response(...) shape:{self.data.shape} kind:{self.kind}'

    def __repr__(self):
        return f'Response(data=<{self.data.shape}>, kind="{self.kind}", scale="{self.scale}")'

    @property
    def dataseries(self) -> dict:
        return {'basename': self.basename,
                'data': self.data,
                'dtype': self.data.dtype,
                'attr': {
                    'class': self.__class__.__name__,
                    'order': self.data.ndim,
                    'kind': self.kind,
                    'scale': self.scale,
                    }
                }


class Spectrum(MyOmdsDatagroupObj):
    basename = 'spectrum'

    def __init__(self, responses=None, axes=None, pols=None):
        super().__init__()
        if responses is None:
            responses = []
        if axes is None:
            axes = []
        if pols is None:
            pols = []

        # should add checking here

        self.responses = responses
        self.axes = axes
        self.pols = pols
        self._attributes['order'] = len(axes)

    @property
    def datagroup(self) -> list:
        return [*self.responses, *self.axes, *self.pols]

    @datagroup.setter
    def datagroup(self, value):
        if not isinstance(value, Iterable):
            value = [value]

        for item in value:
            if isinstance(item, Response):
                self.responses = item
            if isinstance(item, Polarization):
                self.pols = item
            if isinstance(item, Axis):
                self.axes = item

    @property
    def responses(self):
        return self._responses

    @responses.setter
    def responses(self, value):
        if not isinstance(value, Iterable):
            value = [value]

        self._responses = []
        for this_item in value:
            self._responses.append(self._validate_item(this_item, Response))

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, value):
        if not isinstance(value, Iterable):
            value = [value]

        self._axes = []
        for this_item in value:
            self._axes.append(self._validate_item(this_item, Axis))

    @property
    def pols(self):
        return self._pols

    @pols.setter
    def pols(self, value):
        if not isinstance(value, Iterable):
            value = [value]

        self._pols = []
        for this_item in value:
            self._pols.append(self._validate_item(this_item, Polarization))

    @staticmethod
    def _validate_item(item, cls):
        if isinstance(item, cls):
            return item
        else:
            raise ValueError(
                   f"Item {item} is a {item.__class__}. It must be {cls}.")
