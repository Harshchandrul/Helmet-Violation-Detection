# This file will run for only one time after the project has been finish to finalize all the packages required to run the program
from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e.'

def get_requirements(filename:str)->list[str]:
    """
    This Function will return list of requirements from a file
    :param filename: 
    :return: 
    """
    requirements = []
    with open(filename, 'r') as f:
        requirements = f.read().splitlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements


setup(
    name = 'Helmet-Violation-Detection-System',
    version = '0.1',
    author = 'Harsh Chandrul',
    author_email= 'harshchandrul@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)



