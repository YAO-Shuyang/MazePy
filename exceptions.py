
from typing import Optional

class DtypeError(Exception):
    """
    Exception raised when the data type of the array does not satisfy the
    requirement.

    Attributes
    ----------
    dtype : str
        Data type of the array.
    message : str
    """

    def __init__(self, dtype: str, message: Optional[str] = None):
        if message is None:
            message = f"Invalid data type: {dtype}"
        super().__init__(message)

class DimensionError(Exception):
    """
    Exception raised when the dimension of the array does not satisfy the
    requirement.

    Attributes
    ----------
    dim : int
        Dimension of the array.
    message : str
    """

    def __init__(self, ndim: int, message: Optional[str] = None):
        if message is None:
            message = f"Invalid dimension: {ndim}"
        super().__init__(message)
        
class ShapeError(Exception):
    """
    Exception raised when the shape of the array does not satisfy the
    requirement.

    Attributes
    ----------
    shape : int
        Shape of the array.
    message : str
    """

    def __init__(self, shape: tuple, message: Optional[str] = None):
        if message is None:
            message = f"Invalid dimension: {shape}"
        super().__init__(message)

class DimensionMismatchError(Exception):
    """
    Exception raised when two arrays do not have the same length.

    Attributes
    ----------
    len1 : int
        Length of the first array.
    len2 : int
        Length of the second array.
    message : str
        Explanation of the error.
    """

    def __init__(
        self, 
        len1: int, 
        len2: int, 
        message: str = "The lengths of the two arrays do not match."
    ):
        self.message = message
        super().__init__(
            f"{message} First array length: {len1}, Second array length: {len2}"
        )


class ViolatedConstraintError(Exception):
    """
    Exception raised when the constraint is violated.

    Attributes
    ----------
    message : str
        Explanation of the error.
    """

    def __init__(self, message: str):
        super().__init__(message)