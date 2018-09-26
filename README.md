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

SpikeInterface allows the user to extract data from either raw or processed extracellular data with an InputExtractor and OutputExtractor, respectively.

**InputExtrator**

To run our standardized data retrieval functions for your raw extracellular data, import the subclass InputExtractor coinciding with your specific file format. Then you can use that subclass of InputExtractor to extract various information from your file. 

In this example, we assume the user's file format is MountainLab so we will import the MdaInputExtractor.

```python
from spikeinterface import MdaInputExtractor

mie = si.MdaInputExtractor(dataset_directory='kbucket://b5ecdf1474c5/datasets/synth_datasets/datasets/synth_tetrode_30min',
                           download=True)
                           
num_channels = mie.getNumChannels()

print(num_channels)

## [1] 4
```

### Uses


### Future Work
