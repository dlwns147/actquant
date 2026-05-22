from setuptools import setup, find_packages

setup(
    name='common_code',
    version='0.1',
    packages=find_packages(),
    # py_modules=[],
    install_requires=[
        # Add any dependencies your package needs here
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
)