.. spikeextractors documentation master file, created by
   sphinx-quickstart on Sun Jun  2 12:08:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SpikeExtractors's documentation!
===========================================

SpikeExtractors provides tools for extracting, converting between, and curating raw or spike sorted extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and sorted extracellular data.
2. To promote straightforward reuse of extracellular datasets.
3. To increase the reproducibility of electrophysiological studies using spike sorting software.
4. To address issues of file format compatibility within electrophysiology research without creating yet another file format.

SpikeExtractors attempts to standardize data retrieval rather than data storage. This eliminates the need for shared file formats and allows for the creation of new tools built off of our data retrieval guidelines.

In addition to implementing multi-format I/O for various formats, the framework makes it possible to develop software tools that are agnostic to the underlying formats by working with the standardized python objects (recording and sorting extractors). These include processing routines (filters, sorting algorithms, downstream processing), and visualization widgets. It also provides mechanisms for lazy manipulation of recordings and sortings (concatenation, combination, subset extraction).

The following sections will guide you through the basic usage and API for recording extractors, sorting extractors,
available extractors, using a probe file, and curating a sorting output.

Contents:

.. toctree::
   :maxdepth: 3
   
   overview
   recording_extractors
   sorting_extractors
   available_extractors
   probe
   curation  

Contact
=======

If you have questions or comments, please contact Cole Hurwitz: colehurwitz@gmail.com

