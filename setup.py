import setuptools

d = {}
exec(open("spikeextractors/version.py").read(), None, d)
version = d['version']
pkg_name = "spikeextractors"
long_description = open("README.md").read()

setuptools.setup(
    name=pkg_name,
    version=version,
    author="Alessio Buccino, Cole Hurwitz, Samuel Garcia, Jeremy Magland, Matthias Hennig",
    author_email="alessiob@ifi.uio.no",
    description="Python module for extracting recorded and spike sorted extracellular data from different file types and formats",
    url="https://github.com/SpikeInterface/spikeextractors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={},
    include_package_data=True,
    install_requires=[
        'numpy',
        'tqdm',
        'joblib'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
