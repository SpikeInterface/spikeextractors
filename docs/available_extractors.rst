
Supported File Formats
~~~~~~~~~~~~~~~~~~~~

Currently, we support many popular file formats for both raw and sorted extracellular datasets. Given the standardized, modular design of our recording and sorting extractors, adding new file formats is straightforward so we expect this list to grow in future versions.

Raw Data Formats
----------------

For raw data formats, we currently support:

* **Binary** - BinDatRecordingExtractor
* **Biocam HDF5** - BiocamRecordingExtractor
* **Experimental Directory Structure (Exdir)** - ExdirRecordingExtractor
* **Intan** - IntanRecordingExtractor
* **Klusta** - KlustaRecordingExtractor
* **MaxOne** - MaxOneRecordingExtractor
* **Mountainsort MDA** - MdaRecordingExtractor
* **MEArec** - MEArecRecordingExtractor
* **Open Ephys** - OpenEphysRecordingExtractor
* **Phy/Kilosort** - PhyRecordingExtractor/KilosortRecordingExtractor
* **SpikeGLX** - SpikeGLXRecordingExtractor
* **Spyking Circus** - SpykingCircusRecordingExtractor

Sorted Data Formats
-------------------

For sorted data formats, we currently support:

* **Experimental Directory Structure (Exdir)** - ExdirSortingExtractor
* **HerdingSpikes2** - HS2SortingExtractor
* **Kilosort/Kilosort2** - KiloSortSortingExtractor
* **Klusta** - KlustaSortingExtractor
* **Mountainsort MDA** - MdaSortingExtractor
* **MEArec** - MEArecSortingExtractor
* **NPZ (created by SpikeInterface)** - NpzSortingExtractor
* **Open Ephys** - OpenEphysSortingExtractor
* **Spyking Circus** - SpykingCircusSortingExtractor
* **Trideclous** - TridesclousSortingExtractor

We plan to add support for NWB 2.0 in future versions.

Installed Extractors
--------------------

To check which extractors are useable in a given python environment, one can print the installed recording extractor list and the installed sorting extractor list. An example from a newly installed miniconda3 environment is shown below,

First, import the spikeextractors package,

.. code:: python

  import spikeextractors as se

Then you can check the installed RecordingExtractor list,

.. code:: python

  se.installed_recording_extractor_list
  
which outputs,

.. parsed-literal::
  [spikeextractors.extractors.mdaextractors.mdaextractors.MdaRecordingExtractor,
   spikeextractors.extractors.biocamrecordingextractor.biocamrecordingextractor.BiocamRecordingExtractor,
   spikeextractors.extractors.bindatrecordingextractor.bindatrecordingextractor.BinDatRecordingExtractor,
   spikeextractors.extractors.spikeglxrecordingextractor.spikeglxrecordingextractor.SpikeGLXRecordingExtractor,
   spikeextractors.extractors.phyextractors.phyextractors.PhyRecordingExtractor,
   spikeextractors.extractors.maxonerecordingextractor.maxonerecordingextractor.MaxOneRecordingExtractor]
   
and the installed SortingExtractors list,

.. code:: python

  se.installed_sorting_extractor_list

which outputs,

.. parsed-literal::
  [spikeextractors.extractors.mdaextractors.mdaextractors.MdaSortingExtractor,
   spikeextractors.extractors.hs2sortingextractor.hs2sortingextractor.HS2SortingExtractor,
   spikeextractors.extractors.klustasortingextractor.klustasortingextractor.KlustaSortingExtractor,
   spikeextractors.extractors.kilosortsortingextractor.kilosortsortingextractor.KiloSortSortingExtractor,
   spikeextractors.extractors.phyextractors.phyextractors.PhySortingExtractor,
   spikeextractors.extractors.spykingcircussortingextractor.spykingcircussortingextractor.SpykingCircusSortingExtractor,
   spikeextractors.extractors.npzsortingextractor.npzsortingextractor.NpzSortingExtractor]

 
When trying to use an extractor that has not been installed in your environment, an installation message will appear indicating which python packages must be installed as a prerequisite to using the extractor,

.. code:: python

  exdir_file = 'path_to_exdir_file'
  recording = se.ExdirRecordingExtractor(exdir_file)

throws the error,

.. parsed-literal::
  ----> 1 se.ExdirRecordingExtractor(exdir_file)

  ~/spikeextractors/spikeextractors/extractors/exdirextractors/exdirextractors.py in __init__(self, exdir_file)
       22 
       23     def __init__(self, exdir_file):
  ---> 24         assert HAVE_EXDIR, "To use the ExdirExtractors run:\n\n pip install exdir\n\n"
       25         RecordingExtractor.__init__(self)
       26         self._exdir_file = exdir_file

  AssertionError: To use the ExdirExtractors run:

  pip install exdir

So to use either of the Exdir extractors, you must install the python package exdir. The python packages that are required to use of all the extractors can be installed as below,

.. parsed-literal::
  pip install exdir h5py pyintan MEArec pyopenephys tridesclous
