Overview
========

Installation
------------

To get started with spikeextractors, you can install it with pip:

.. parsed-literal::
  pip install spikeextractors

To get updated versions, periodically run:

.. parsed-literal::
  pip install --upgrade spikeextractors

You can also install spikeextractors locally by cloning the repo to your local machine and then installing with setup.py,

.. parsed-literal::
  git clone https://github.com/SpikeInterface/spikeextractors.git
  cd spikeextractors
  python setup.py install

Tools using SpikeExtractors
---------------------------

`spiketoolkit 
<https://github.com/SpikeInterface/spiketoolkit>`_
- A repository containing tools for analysis and evaluation of extracellular recordings built with spikeextractors.  It also contains wrapped spike sorting algorithms that take in recording extractors and output sorting extractors, allowing for standardized evaluation and quality control.

`spikewidgets 
<https://github.com/SpikeInterface/spikewidgets>`_
- A repository containing graphical widgets built with spikeextractors to visualize both the raw and sorted extracellular data along with sorting results. 

`spikeforest 
<https://github.com/flatironinstitute/spikeforest>`_
- SpikeForest is a continuously updating platform which benchmarks the performance of spike sorting codes across a large curated database of electrophysiological recordings with ground truth.

`spikely 
<https://github.com/rogerhurwitz/spikely>`_
- An application for processing extracellular data that utilizes both spikeextractors and spiketoolkit. This application can run any supported spike sorting algorithm on extracellular data that is stored in any supported file format.

Contact
-------

If you have questions or comments, please contact Cole Hurwitz: colehurwitz@gmail.com
