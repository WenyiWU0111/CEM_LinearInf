import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CEM_LinearInf",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="WenyiWU0111",                     # Full name of the author
    description="CEM-LinearInf is a Python package for linear causal inference, which can help you implement the whole process of causal inference easily.",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    #py_modules=["CEM_LinearInf"],             # Name of the python package
    #package_dir={'':'CEM_LinearInf'},     # Directory of the source code of the package
    install_requires=['numpy', 'pandas', 'scipy>=1.11.1', 'scikit-learn', 'statsmodels', 'matplotlib', 'seaborn']                     # Install other dependencies if any
)
