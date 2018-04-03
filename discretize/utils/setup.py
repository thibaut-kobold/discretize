import os
import os.path
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('utils', parent_package, top_path)
    return config


if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
