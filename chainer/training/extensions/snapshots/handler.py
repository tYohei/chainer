import h5py
import numpy

from chainer.serializers import npz


def save_npz(filename, target, compression=True):
    with open(filename, 'wb') as f:
        if compression:
            numpy.savez_compressed(f, **target)
        else:
            numpy.savez(f, **target)


def save_hdf5(filename, target, compression=4):
    with h5py.File(filename, 'w') as f:
        for key, value in target.items():
            key = '/' + key.lstrip('/')
            f.create_dataset(key, data=value,
                             compression=compression,
                             dtype=h5py.special_dtype(vlen=unicode))


class SerializerHandler(object):
    """Base class of handler of serializers.

    This handler is used in snapshot extension in :class:`Trainer`.
    To divide the timinig of serialization and actual saving, this class
    provides :func:`serialize()` and :func:`save` functions.
    """

    def __init__(self, savefun=save_npz, **kwds):
        self.savefun = savefun
        self.kwds = kwds

    def serialize(self, target):
        """Serialize the given target.

        This method creates a standard serializer in Chainer and serialize
        a target using this serializer.

        Args:
            target: Object to be serialize. Usually it is a trainer object.
        """
        self.serializer = npz.DictionarySerializer()
        self.serializer.save(target)
        self.target = self.serializer.target

    def save(self, filename):
        """Save the serialized target with a given file name.

        This method actually saves the registered serialized target with a
        given file name.
        """
        self.savefun(filename, self.target, **self.kwds)
