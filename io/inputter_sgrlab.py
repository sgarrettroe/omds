from . import Inputter, logger
from ..base import (MyOmdsDatagroupObj, MyOmdsDataseriesObj, Spectrum,
                    Response, Polarization, Axis, UNITS)
from scipy.io import loadmat
import numpy as np

REQUIRED_FIELDS = [
    'w1',
    't2',
    'w3',
    'R',
    'time_units',
    'freq_units',
]


class InputterSGRLab(Inputter):
    """Class to import from SGRLab matlab data structure in a .mat file.
    """
    def __init__(self):
        super().__init__()
        pass

    def input(self, file_name: str) -> list:
        mat_dict = loadmat(file_name, simplify_cells=True)
        data_list = [self.process_item(k, v) for k, v in mat_dict.items()
                     if self.process_item(item) is not None]
        return data_list

    def process_item(self, key, value) -> (list | MyOmdsDataseriesObj |
                                           MyOmdsDatagroupObj | None):
        logger.debug(f'processing variable {key} in .mat file')
        if not isinstance(value, list):
            item = [value]
        else:
            item = value

        for this_item in item:
            # test if all fields are in item
            if isinstance(this_item, dict):

                for field in REQUIRED_FIELDS:
                    if field not in item:
                        logger.debug(f'{key} not an SGRLab 2D-IR data '
                                     f'structure')
                        return None

                #
                # extract information
                #
                n1, n3 = this_item[0]['R'].shape
                n2 = len(this_item)
                R = np.zeros(shape=(n1, n2, n3))
                for ii in range(len(this_item)):
                    R[:, ii, :] = this_item[ii]['R']
                resp = Response(R, kind='absorptive', scale='mOD')

                polarization = this_item[0]['polarization']
                pols = []
                for p in polarization:
                    pols.append(Polarization(p.upper()))

                w1 = this_item[0]['w1']
                t2 = np.array([d['t2'] for d in this_item])
                w3 = this_item[0]['w3']

                tu = this_item[0].time_units
                fu = this_item[0].freq_units

                #
                # pack it up
                #
                opt = {'fftshift': True}
                x1 = Axis(w1, units=UNITS.__getitem__(fu.upper()), options=opt)
                x2 = Axis(t2, units=UNITS.__getitem__(tu.upper()))
                x3 = Axis(w3, units=UNITS.__getitem__(fu.upper()))

                return Spectrum(responses=resp, pols=pols, axes=[x1, x2, x3])
