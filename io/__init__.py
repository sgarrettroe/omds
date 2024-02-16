import h5py
import logging

# set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s'
)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


class Outputter:
    """Base class for output functionality. All output methods should
    inherit from this class.
    """
    # take response function in and provide fxn to write output file

    def output(self, obj, filename, root='/', access_mode='w',
               scidata_iri='') -> None:
        pass


class Inputter:
    """Base class for all input classes.

    """
    def __init__(self):
        pass

    def input(self, file_name: str) -> list:
        raise NotImplementedError("Please implement this import function.")


def myh5disp(group):
    def helper(name, obj):
        if isinstance(obj, h5py.Group):
            print(name)
        elif isinstance(obj, h5py.Dataset):
            print(name, obj.dtype, obj.shape)
    group.visititems(helper)
