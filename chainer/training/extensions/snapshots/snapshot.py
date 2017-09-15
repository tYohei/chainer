from chainer.training import extension
from chainer.training.extensions.snapshot import condition as condition_module
from chainer.training.extensions.snapshot import handler as handler_module
from chainer.training.extensions.snapshot import writer as writer_module


class Snapshot(extension.Extension):
    """Takes a snapshot.

    Args:
        target: Object to serialize. If not specified, it will
            be trainer object.
        condition: Condition object. It must be a callable object that
            returns boolean in its call. If its returns True the snapshot will
            be done. If not it will be skipped.
        writer: Writer object. It need to be a callable object.
        handler: Serializer handler object.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.

    """

    def __init__(self,
                 target=None,
                 condition=condition_module.Always(),
                 writer=writer_module.SimpleWriter(),
                 handler=handler_module.SerializerHandler(),
                 filename='snapshot_iter_{.updater.iteration}'):
        self._target = target
        self._filename = filename
        self.condition = condition
        self.writer = writer
        self.handler = handler

    def __call__(self, trainer):
        if self.condition(trainer, self):
            target = trainer if self._target is None else self._target
            self.handler.serialize(target)
            filename = self._filename
            if callable(filename):
                filename = filename(trainer)
            filename = filename.format(trainer)
            outdir = trainer.out
            self.writer(filename, outdir, self.handler)

    def finalize(self):
        if hasattr(self.writer, 'finalize'):
            self.writer.finalize()
