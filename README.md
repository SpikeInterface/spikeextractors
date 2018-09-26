# SpikeInterface

SpikeInterface is a module that enables easy creation and deployment of tools for extracting extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and processed extracellular data.
2. To promote straightfoward reuse of extracellular datasets.
3. To increase reproducibility of electrophysiological studies using spike sorting software.

Traditionally, researchers have attempted to achieve the above goals by creating standardized file formats for extracellular data. Although this approach seems promising, it can run into issues with legacy data and software, the need for wide-scale adoption of the format, steep learning curves, and an inability to adapt to new storage needs from experimental labs.

To circumvent these problems, we designed SpikeInterface -- a module that attempts to standardize *data retrieval* rather than data storage. By standardizing data retrieval from extracellular datasets, we can eliminate the need for shared file formats and allow for the creation of new tools built off of our data retrieval guidelines.

## Getting Started with Preexisting Code

To get started with SpikeInterface, clone the repo into your code base.

```shell
https://github.com/colehurwitz31/spikeinterface.git
```

SpikeInterface allows the user to extract data from either raw or processed extracellular data with an InputExtractor or OutputExtractor, respectively.


**InputExtrator**

To run our standardized data retrieval functions for your raw extracellular data, import the subclass InputExtractor coinciding with your specific file format. Then, you can use that subclass of InputExtractor to extract data and information from your raw data file. 

In this example, we assume the user's raw file format is MountainLab so we will import the MdaInputExtractor.

```python
from spikeinterface import MdaInputExtractor
dataset_directory_path = 'kbucket://b5ecdf1474c5/datasets/synth_datasets/datasets/synth_tetrode_30min'

mie = si.MdaInputExtractor(dataset_directory=dataset_directory_path, download=True)
                           
print(mie.getNumChannels())

## Out[1] 4
```

**OutputExtractor**

To run our standardized data retrieval functions for your processed extracellular data, import the subclass OutputExtractor coinciding with your specific file format/spike sorter. Then, you can use that subclass of OutputExtractor to extract data and information from your processed data file. 

In this example, we assume the user's processed file format is also MountainLab so we will import the MdaOutputExtractor.

```python
from spikeinterface import MdaOutputExtractor
firings_file_path = 'kbucket://b5ecdf1474c5/datasets/synth_datasets/datasets/synth_tetrode_30min/firings_true.mda'

moe = si.MdaOutputExtractor(firings_file=firings_file_path)
                           
print(moe.getUnitSpikeTrain(unit_id=0)

## Out[3]:array([  2.71249481e+03,   1.22188979e+04,   1.83042929e+04, ...,
##              5.39305688e+07,   5.39829415e+07,   5.39836896e+07])
```

### Uses


### Future Work
