[![Build Status](https://travis-ci.org/SpikeInterface/spikeextractors.svg?branch=master)](https://travis-ci.org/colehurwitz31/spikeextractors)

Alpha Development
Version 0.4.4


# spikeextractors

SpikeExtractors provides tools for extracting, converting between, and curating raw or spike sorted extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and sorted extracellular data.
2. To promote straightfoward reuse of extracellular datasets.
3. To increase reproducibility of electrophysiological studies using spike sorting software.
4. To address issues of file format compatibility within electrophysiology research without creating yet another file format.

SpikeExtractors attempts to standardize *data retrieval* rather than data storage. This eliminates the need for shared file formats and allows for the creation of new tools built off of our data retrieval guidelines.

In addition to implementing multi-format I/O for various formats, the framework makes it possible to develop software tools that are agnostic to the underlying formats by working with the standardized python objects (recording and sorting extractors). These include processing routines (filters, sorting algorithms, downstream processing) and visualization widgets. It also provides mechanisms for lazy manipulation of recordings and sortings (concatenation, combination, subset extraction).

## Installation

To get started with spikeextractors, you can install it with pip:

```shell
pip install spikeextractors
```

To get updated versions, periodically run:

```shell
pip install --upgrade spikeextractors
```

You can also install spikeextractors locally in development mode by cloning the repo to your local machine and then installing in editable mode.

```shell
git clone https://github.com/SpikeInterface/spikeextractors.git

cd spikeextractors
pip install -e .
```

## Getting Started with Preexisting Code

There are two typies of spike extractors: recording extractors (inherited from RecordingExtractor) and sorting extractors (inherited from SortingExtractor). These allow the user to represent data from multi-channel raw extracellular traces (recordings) and the results of spike sorting (sortings).


**RecordingExtractor**

To work with raw extracellular data, import the subclass of RecordingExtractor coinciding with your specific file format. Then, you can use an instance of that class to extract data snippets and information from your raw data file. 

In this [example](https://github.com/SpikeInterface/spikeextractors/blob/master/examples/getting_started_with_recording_extractors.ipynb) we show how to use a RecordingExtractor subclass on a generated, pure-noise timeseries dataset and a linear probe geometry.

First we will generate the properties, data, and probe geometry for this pure-noise dataset. 

```python
# Properties of the in-memory dataset
num_channels=7
samplerate=30000
duration=20
num_timepoints=int(samplerate*duration)

# Generate a pure-noise timeseries dataset and a linear geometry
timeseries=np.random.normal(0,10,(num_channels,num_timepoints))
geom=np.zeros((num_channels,2))
geom[:,0]=range(num_channels)
```

Now we can import spikeextractors and use the NumpyRecordingExtractor since the raw data was stored in the numpy array format. (Typically the data would originate from a file on disk, but we are using an in-memory dataset for illustration.)

```python
from spikeextractors import se

# Define the in-memory recording extractor
RX=se.NumpyRecordingExtractor(timeseries=timeseries,geom=geom,samplerate=samplerate)
```

You can use the RecordingExtractor to retrieve data and information from the dataset with a variety of standard functions that are predefined in the RecordingExtractor base class.

```python
print('Num. channels = {}'.format(len(RX.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(RX.get_sampling_frequency()))
print('Num. timepoints = {}'.format(RX.get_num_frames()))
print('Stdev. on third channel = {}'.format(np.std(RX.get_traces(channel_ids=2))))
print('Location of third electrode = {}'.format(RX.get_channel_property(channel_id=2, property_name='location')))
```
```output
Num. channels = 7
Sampling frequency = 30000 Hz
Num. timepoints = 600000
Stdev. on third channel = 9.99206377601932
Location of third electrode = [ 2.  0.]
```

RecordingExtractor subclasses also provide functionality to save the raw data with the specific format for which the RecordingExtractor was implemented. 

We will now convert our numpy data into the MountainSort format and save it with a MountainSort RecordingExtractor and our previously defined RecordingExtractor.

```python
# Write this dataset in the MountainSort format
si.MdaRecordingExtractor.write_recording(recording=RX,save_path='sample_mountainsort_dataset')
```

The modular design of RecordingExtractor allow them to be used in a variety of other tasks. For example, RecordingExtractors can extract subsets of data from a raw data file or can extract data from multiple files with SubRecordingExtractors and MultiRecordingExtractors. 

**SortingExtractor**

To run our standardized data retrieval functions for your sorted extracellular data, import the subclass SortingExtractor coinciding with your specific file format/spike sorter. Then, you can use that subclass of SortingExtractor to extract data and information from your spike sorted data file. We will show the functionality of the SortingExtractor by continuing our previous example. 

First, we will add some random events and then use the NumpySortingExtractor to extract data about these events. Generally, SortingExtractors would be instantiated with a path the file containing information about the spike sorted units, but since this is a self-contained [example](https://github.com/SpikeInterface/spikeextractors/blob/master/examples/getting_started_with_sorting_extractors.ipynb), we will add the units manually to the SortingExtractor and show how to use it afterwards.

```python
# Generate some random events
times=np.sort(np.random.uniform(0,num_timepoints,num_events))
labels=np.random.randint(1,num_units+1,size=num_events)
    
# Define the in-memory output extractor
SX=si.NumpySortingExtractor()
for k in range(1,num_units+1):
    times_k=times[np.where(labels==k)[0]]
    SX.add_unit(unit_id=k,times=times_k)
```
<br/>

Now, we will demonstrate the API for extracting information from the sorted data using standardized functions from the SortingExtractor.

```python
print('Unit ids = {}'.format(SX.get_unit_ids()))
st=SX.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1=SX.get_unit_spike_train(unit_id=1,start_frame=0,end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
```
```output
Unit ids = [1, 2, 3, 4]
Num. events for unit 1 = 262
Num. events for first second of unit 1 = 8
```

Finally, we can write out our sorted file to the MountainSort format by using the built-in write_sorting method in the MountainSort SortingExtractor subclass.
```python
se.MdaSortingExtractor.write_sorting(sorting=SX,save_path='sample_mountainsort_dataset/firings_true.mda')
```

Now that we have written out our numpy recorded and sorted files in the the MountainSort format, we can easily use the MdaRecordingExtractor and MdaSortingExtractor for our new datasets and the functionality sould be the same.

```python
# Read the raw and sorted datasets with the Mda recording and sorting extractor static methods
RX2=se.MdaRecordingExtractor(dataset_directory='sample_mountainsort_dataset')
SX2=se.MdaSortingExtractor(firings_file='sample_mountainsort_dataset/firings_true.mda')

# We should get he same information as above
print('Unit ids = {}'.format(SX2.get_unit_ids()))
st=SX2.get_unit_spike_train(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1=SX2.get_unit_spike_train(unit_id=1,start_frame=0,end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
```
```output
Unit ids = [1 2 3 4]
Num. events for unit 1 = 262
Num. events for first second of unit 1 = 8
```
SortingExtractors can also extract subsets of data from a sorted data file or can extract data from multiple files with SubSortingExtractor and MultiSortingExtractor, respectively.

This concludes the basic tutorial about the Recording/Sorting Extractors. To see currently implemented extractor subclasses, please check the [extractors](https://github.com/SpikeInterface/spikeextractors/tree/master/spikeextractors/extractors) folder in our repo. 

We have also implemented a variety of tools which use RecordingExtractors and SortingExtractors. Links to these tools are contained in the **Tools that use RecordingExtractors and SortingExtractors** section of the README.

<br/>

## Building a new RecordingExtractor/SortingExtractor

Building a new RecordingExtractors or SortingExtractors for specific file format is as simple as creating a new subclass based on the predefined base classes provided in spikeextractors.

To enable standardization among subclasses, RecordingExtractor and SortingExtractor are abstract base classes which require a new subclass to override all methods which are decorated with @abstractmethod.

An example of how a new subclass for SortingExtractor can be created is provided below.

```python
from spikeextractors import SortingExtractor

class ExampleSortingExtractor(SortingExtractor):
    def __init__(self, ex_parameter_1, ex_parameter_2):
        SortingExtractor.__init__(self)
        
        ## All file specific initialization code can go here.
        
    def get_unit_ids(self):
    
        #Fill code to get a unit_ids list containing all the ids (ints) of detected units in the recording
        
        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
    
        '''Code to extract spike frames from the specified unit.
        It will return spike frames from within three ranges:
            [start_frame, t_start+1, ..., end_frame-1]
            [start_frame, start_frame+1, ..., final_unit_spike_frame - 1]
            [0, 1, ..., end_frame-1]
            [0, 1, ..., final_unit_spike_frame - 1]
        if both start_frame and end_frame are given, if only start_frame is
        given, if only end_frame is given, or if neither start_frame or end_frame
        are given, respectively. Spike frames are returned in the form of an
        array_like of spike frames. In this implementation, start_frame is inclusive
        and end_frame is exclusive conforming to numpy standards.
       
        '''
        
        return spike_train
    
    .
    .
    .
    . 
    . #Optional functions and pre-implemented functions that a new SortingExtractor doesn't need to implement
    .
    .
    .
    .
        
    @staticmethod
    def write_sorting(sorting, save_path):
        '''
        This is an example of a function that is not abstract so it is optional if you want to override it. It allows other 
        SortingExtractors to use your new SortingExtractor to convert their sorted data into your 
        sorting file format.
        '''
```

As you can see, our extractor base classes were designed to make implementing a new subclass as straightforward and flexible as possible while still enforcing standardized data retrieval functions.

Once all abstract methods are overwritten in your RecordingExtractor or SortingExtractor, your subclass is ready for deployment and can be used with any pre-implemented tools (see **Tools that use RecordingExtractors and SortingExtractors**).
<br/>

## Interactive Example

To experiment with RecordingExtractors and SortingExtractors in a pre-installed environment, we have provided a [Collaborative environment](https://gist.github.com/magland/e43542fe2dfe856fd04903b9ff1f8e4e). If you click on the link and then click on "Open in Collab", you can run the notebook and try out the features of and tools for spikeextractors.
<br/>

## Tools that use RecordingExtractors and SortingExtractors

- [spiketoolkit](https://github.com/SpikeInterface/spiketoolkit) - A repository containing tools for analysis and evaluation of extracellular recordings built with spikeextractors.  It also contains wrapped spike sorting algorithms that take in recording extractors and output sorting extractors, allowing for standardized evaluation and quality control.
- [spikewidgets](https://github.com/SpikeInterface/spikewidgets) - A repository containing graphical widgets built with spikeextractors to visualize both the raw and sorted extracellular data along with sorting results. 

## Authors

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland 

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Mathematics (CCM), Flatiron Institute, New York, United States

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland
<br/>
<br/>
For any correspondence, contact Cole Hurwitz at colehurwitz@gmail.com

## Contributors

[Samuel Garcia](https://github.com/samuelgarcia) - Centre de Recherche en Neuroscience de Lyon (CRNL), Lyon, France
