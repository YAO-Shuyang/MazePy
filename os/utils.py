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

import yaml
def load_yaml(yaml_dir: str) -> dict:
    """load_yaml: safely load yaml which contains python tuple. If you directly use yaml.safe_load(), it may raise 
                  an error:
                    "yaml.constructor.ConstructorError: could not determine a constructor for the tag 'tag:yaml.org,2002:python/tuple'"
                  So, here we add this tag.

    Parameters
    ----------
    yaml_dir : str
        The directory of the *.yaml file to be opened.

    Returns
    -------
    dict
        The dictionary content saved in the yaml file.
    """ 
    # Define the custom constructor for Python tuples
    def construct_tuple(loader: yaml.Loader, node:yaml.Node):
        return tuple(loader.construct_sequence(node))

    # Create a custom loader by inheriting from SafeLoader
    class CustomLoader(yaml.SafeLoader):
        pass

    # Add the custom constructor to the CustomLoader
    CustomLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_tuple)

    # Read the YAML file using the CustomLoader
    with open(yaml_dir, 'r') as file:
        yaml_data = yaml.load(file, Loader=CustomLoader)

    return yaml_data
