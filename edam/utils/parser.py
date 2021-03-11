from typing import IO, Any, TypeVar

import numpy as np


_T = TypeVar("_T")

def txt_to_nparray(file: IO[Any], **kwargs) -> np.ndarray:
    """Converts txt file into numpy array.

    Examples:
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> txt_to_nparray(c)
        array([[0., 1.],
            [2., 3.]])
    Arguments:
        file {io.TextIOWrapper} -- Txt input file.

    Returns:
        np.ndarray -- [description]
    """
    return np.loadtxt(file, **kwargs)
