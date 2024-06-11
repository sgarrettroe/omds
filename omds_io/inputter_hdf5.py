from . import Inputter, logger
from base import MyOmdsDatagroupObj, MyOmdsDataseriesObj, Spectrum, Response, Polarization, Axis
import h5py


class InputterHDF5(Inputter):
    """Class to import from HDF5.
    """
    def __init__(self):
        super().__init__()
        pass

    def input(self, file_name: str) -> list:
        with h5py.File(file_name, 'r') as f:
            data_list = [self.process_item(item) for item in f.values()]
        return data_list


    def process_item(self, item) -> list | MyOmdsDataseriesObj | MyOmdsDatagroupObj:

        if isinstance(item, h5py.Group):
            if item.name == '/':
                logger.debug('found root group')
                return [self.process_item(this_item) for this_item in item.values()]

            else:
                if 'class' not in item.attrs.keys():
                    logger.warning('Folders like raw/ not yet implemented. Skipping...')
                    return [item.name]

                elif item.attrs['class'] == MyOmdsDatagroupObj.__name__:
                    logger.debug('found datagroup')
                    return MyOmdsDatagroupObj([self.process_item(this_item)
                                               for this_item in item.values()])
                elif item.attrs['class'] == Spectrum.__name__:
                    logger.debug('found Spectrum')
                    return Spectrum(responses=[self.process_response_item(this_item)
                                               for this_item in item.values()
                                               if this_item.attr['class']==Response.__name__],
                                    pols=[self.process_polarization_item(this_item)
                                               for this_item in item.values()
                                               if this_item.attr[
                                                   'class'] == Polarization.__name__],
                                    axes=[self.process_axis_item(this_item)
                                               for this_item in item.values()
                                               if this_item.attr[
                                                   'class'] == Axis.__name__],
                                    )
                else:
                    raise TypeError(f'Expected datagroup | dataseries, found {item.attrs["class"]}')

        if isinstance(item, h5py.Dataset):
            if item.attrs['class'] == MyOmdsDataseriesObj.__name__:
                raise NotImplementedError('No default data series implemented yet.')
            else:
                logger.debug('found a dataseries in root group')
                if item.attrs['class'] == Response.__name__:
                    return self.process_response_item(item)
                elif item.attrs['class'] == Polarization.__name__:
                    return self.process_polarization_item(item)
                elif item.attrs['class'] == Axis.__name__:
                    return self.process_axis_item(item)
                else:
                    raise TypeError(f'Unknown data class found {item.attrs["class"]}')

    def process_response_item(self, item):
        logger.debug(f'Found response item {item.name}')
        return Response(data=item[()], kind=item.attrs['kind'],scale=item.attrs['scale'])

    def process_polarization_item(self, item):
        logger.debug(f'Found polarization item {item.name}')
        return Polarization(pol_in=item.attrs['label'])

    def process_axis_item(self, item):
        logger.debug(f'Found axis item {item.name}')
        return Axis(x=item, units=item.attrs['units'])
