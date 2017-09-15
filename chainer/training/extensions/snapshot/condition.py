class Condition(object):
    """Base class of snapshot condition.

    This class gives the condition if a snapshot should be taken or not.
    :func:`__call__()` is invoked every time when :class:`Snapshot` object's
    extension trigger is pulled.
    """

    def __init__(self):
        pass

    def __call__(self, trainer, snapshot):
        """Determine the condition is met or not.

        Args:
            trainer (Trainer): Trainer object that invokes this operator
                indirectly.
            snapshot (Snapshot): Snapshot object that invokes this operator
                directly.

        Returns:
            bool: True if condition met else false.
        """
        return False


class Always(Condition):
    """Snapshot condition that always return true.

    This class always returns true for its condition. This is the default
    condition for :class:`Snapshot` object.
    """

    def __init__(self):
        pass

    def __call__(self, trainer, snapshot):
        return True
