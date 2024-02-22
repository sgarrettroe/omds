import os
import re

import h5py

from . import Outputter
from . import MyOmdsDataseriesObj, MyOmdsDatagroupObj, Spectrum, Response, Axis, Polarization


class OutputterHDF5(Outputter):
    """Class to write output to an HDF5 file.
    """
    # default output mechanism, writes HDF5 file
    def output(self, obj_in, filename, root='/', access_mode='w',
               scidata_iri='') -> None:
        """Output HDF5 file that saves the dataseries or sets of them.

        Parameters
        ----------
        obj_in : MyOmdsDataseriesObj or list of MyOmdsDataseriesObj
            The input objects to be processed. They must have an
            attribute dataseries. Dataseries must be a dictionary with
            keys "basename", "data", and "dtype".
        filename : str
            The name of the output file.
        root :
            The base group of the dataseries(s).
        access_mode : {'w','a'}, optional
            The mode used to open the output file. The default is 'w',
            which overwrites and existing file.
        scidata_iri : str, optional
            A link back to the scidata description (usually a json-ld
            file) that points to this datagroup or dataset.

        Returns
        -------
        None
        """

        # make sure there is one trailing / in the root name
        root = root.rstrip('/') + '/'

        def attach_axis_to_responses(axis_dset: h5py.Dataset,
                                     current_hdf_group: h5py.Group) -> None:
            """Attach the axis object to the responses found in the
            current group.

            References
            ---
            https://docs.h5py.org/en/stable/high/dims.html
            """
            axis_dset.make_scale()

            # get the number of the axis
            m = re.match(rf'{Axis.basename}(\d+)', os.path.basename(axis_dset.name))
            if m:
                dim_idx = int(m.group(1)) - 1
            else:
                raise ValueError(f'Axis name {axis_dset.name} does not match '
                                 f'required pattern {Axis.basename}<int>.')

            name = Response.basename
            r_idx = 1
            flag_continue = True
            while flag_continue:
                r_name = f'{name}{r_idx}'
                if current_hdf_group.__contains__(f'{r_name}'):
                    current_hdf_group[r_name].dims[dim_idx].attach_scale(axis_dset)
                    r_idx += 1
                else:
                    flag_continue = False

        def process_item(obj: list | MyOmdsDataseriesObj | MyOmdsDatagroupObj,
                         grp):
            if isinstance(obj, list):
                for this_obj in obj:
                    process_item(this_obj, grp)

            elif isinstance(obj, MyOmdsDatagroupObj):
                idx = 1
                while grp.__contains__(f'{obj.basename}{idx}'):
                    idx += 1

                sub_grp = grp.create_group(f'{obj.basename}{idx}')
                for (key, val) in obj.attributes.items():
                    sub_grp.attrs[key] = val

                for this_obj in obj:
                    process_item(this_obj, sub_grp)

            elif isinstance(obj, MyOmdsDataseriesObj):
                dset = obj.dataseries
                idx = 1
                while grp.__contains__(f'{obj.basename}{idx}'):
                    idx += 1
                h5dset = grp.create_dataset(f'{obj.basename}{idx}',
                                            dset['data'].shape,
                                            dtype=dset['dtype'],
                                            data=dset['data'])
                for (key, val) in dset['attr'].items():
                    h5dset.attrs[key] = val

                # attach "dimension scales" for Axis objects
                if isinstance(obj, Axis):
                    attach_axis_to_responses(h5dset, grp)
            else:
                raise TypeError(f'Object has incorrect type. Expected'
                                f'list | MyOmdsDataseries | MyOmdsDatagroup, '
                                f'found {type(obj)}.')

        with h5py.File(filename, access_mode) as f:
            if f.__contains__(f'{root}'):
                grp = f
            else:
                grp = f.create_group(root)
            if root == '/':
                f.attrs['scidata_iri'] = scidata_iri
            process_item(obj_in, grp)  # recursively process input
