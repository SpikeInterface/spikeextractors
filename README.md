[![Build Status](https://travis-ci.org/SpikeInterface/spikeextractors.svg?branch=master)](https://travis-ci.org/SpikeInterface/spikeextractors) [![PyPI version](https://badge.fury.io/py/spikeextractors.svg)](https://badge.fury.io/py/spikeextractors)

Alpha Development

# SpikeExtractors

SpikeExtractors provides tools for extracting, converting between, and curating raw or spike sorted extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and sorted extracellular data.
2. To promote straightforward reuse of extracellular datasets.
3. To increase the reproducibility of electrophysiological studies using spike sorting software.
4. To address issues of file format compatibility within electrophysiology research without creating yet another file format.

SpikeExtractors attempts to standardize *data retrieval* rather than data storage. This eliminates the need for shared file formats and allows for the creation of new tools built off of our data retrieval guidelines.

In addition to implementing multi-format I/O for various formats, the framework makes it possible to develop software tools that are agnostic to the underlying formats by working with the standardized python objects (recording and sorting extractors). These include processing routines (filters, sorting algorithms, downstream processing), and visualization widgets. It also provides mechanisms for lazy manipulation of recordings and sortings (concatenation, combination, subset extraction).

## Installation

To get started with SpikeExtractors, you can install it with pip:

```shell
pip install spikeextractors
```

You can also get SpikeExtractors through the [spikeinterface](https://github.com/SpikeInterface/spikeinterface) package:

```shell
pip install spikeinterface
```

To get updated versions, periodically run:

```shell
pip install --upgrade spikeextractors
```

You can also install SpikeExtractors locally by cloning the repo into your code base. If you install SpikeToolkit locally, you need to run the setup.py file.

```shell
git clone https://github.com/SpikeInterface/spikeextractors.git

cd spikeextractors
python setup.py install
```

## Documentation

The documentation page for the SpikeInterface project can be found here: https://spikeinterface.readthedocs.io/en/latest/

## Authors

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Mathematics (CCM), Flatiron Institute, New York, United States

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland

[Samuel Garcia](https://github.com/samuelgarcia) - Centre de Recherche en Neuroscience de Lyon (CRNL), Lyon, France

For any correspondence, contact Cole Hurwitz at colehurwitz@gmail.com or create a new issue above.
