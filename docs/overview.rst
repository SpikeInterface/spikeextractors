Overview
========

SpikeExtractors provides tools for extracting, converting between, and curating raw or spike sorted extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and sorted extracellular data.
2. To promote straightforward reuse of extracellular datasets.
3. To increase the reproducibility of electrophysiological studies using spike sorting software.
4. To address issues of file format compatibility within electrophysiology research without creating yet another file format.

SpikeExtractors attempts to standardize data retrieval rather than data storage. This eliminates the need for shared file formats and allows for the creation of new tools built off of our data retrieval guidelines.

In addition to implementing multi-format I/O for various formats, the framework makes it possible to develop software tools that are agnostic to the underlying formats by working with the standardized python objects (recording and sorting extractors). These include processing routines (filters, sorting algorithms, downstream processing), and visualization widgets. It also provides mechanisms for lazy manipulation of recordings and sortings (concatenation, combination, subset extraction).

The following sections will guide you through the basic usage and API for recording extractors, sorting extractors,
available extractors, using a probe file, and curating a sorting output.

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
