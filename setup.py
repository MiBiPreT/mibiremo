from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mibiremo',
    version='0.1.0',
    python_requires='>=3.10.0',
    description='MiBiReMo: reaction module based on PhreeqcRM for hydrogeological models',
    long_description=readme,
    author='Matteo Masi',
    author_email='matteo@dndbiotech.it',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs','examples')),
    package_data = {'mibiremo': ['database/*', 'lib/*']},
    include_package_data=True
)

