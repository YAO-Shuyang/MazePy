import os
from os.path import exists, join

def mkdir(path: str) -> bool:
    """mkdir: make directory

    Parameters
    ----------
    path : str
        The directory that to be made up. It could be a file or a folder.

    Returns
    -------
    bool
        Whether successfully make a new folder/file.
    """
    path=path.strip()
    path=path.rstrip("\\")

    if not exists(path):
        os.makedirs(path)
        print("        "+path + ' is made up successfully!')
        return True
    else:
        print("        "+path + ' is already existed!')
        return False