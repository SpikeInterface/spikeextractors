# SpikeInterface

SpikeInterface is a module that enables easy creation and deployment of tools for extracting extracellular data from any file format. Its design goals are as follows:

1. To facilitate standardized analysis and visualization for both raw and processed extracellular data.
2. To promote straightfoward reuse of extracellular datasets.
3. To increase reproducibility of electrophysiological studies using spike sorting software.

Traditionally, researchers have attempted to achieve the above goals by creating standardized file formats for extracellular data. Although this approach seems promising, it can run into issues with legacy data and software, the need for wide-scale adoption of the format, steep learning curves, and an inability to adapt to new storage needs from experimental labs.

To circumvent these problems, we designed SpikeInterface -- a module that attempts to standardize *data retrieval* rather than data storage. By standardizing data retrieval from extracellular datasets, we can eliminate the need for shared file formats and allow for the creation of new tools built off of our data retrieval guidelines.
<br/>
<br/>
## Getting Started with Preexisting Code

To get started with SpikeInterface, clone the repo into your code base.

```shell
https://github.com/colehurwitz31/spikeinterface.git
```

SpikeInterface allows the user to extract data from either raw or spike sorted datasets with an InputExtractor or OutputExtractor, respectively.


**InputExtractor**

To run our standardized data retrieval functions for your raw extracellular data, import the subclass InputExtractor coinciding with your specific file format. Then, you can use that subclass of InputExtractor to extract data snippets and information from your raw data file. 

In this [example](https://github.com/colehurwitz31/spikeinterface/blob/master/examples/getting_started_with_input_extractors.ipynb) from the examples repo, we show how to use an InputExtractor subclass on a generated, pure-noise timeseries dataset and a linear probe geometry.

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

Now we can import SpikeInterface and use the NumpyInputExtractor since the raw data was stored in the numpy array format.

```python
from spikeinterface import si

# Define the in-memory input extractor
IX=si.NumpyInputExtractor(timeseries=timeseries,geom=geom,samplerate=samplerate)
```

Now we can use the InputExtractor to retrieve data and information from the dataset with a variety of standard functions that are predefined in the InputExtractor base class.

```python
print('Num. channels = {}'.format(IX.getNumChannels()))
print('Sampling frequency = {} Hz'.format(IX.getSamplingFrequency()))
print('Num. timepoints = {}'.format(IX.getNumFrames()))
print('Stdev. on third channel = {}'.format(np.std(IX.getRawTraces(channel_ids=2))))
print('Location of third electrode = {}'.format(IX.getChannelInfo(channel_id=2)['location']))IX=si.NumpyInputExtractor(timeseries=timeseries,geom=geom,samplerate=samplerate)



OUT[1] Num. channels = 7
OUT[2] Sampling frequency = 30000 Hz
OUT[3] Num. timepoints = 600000
OUT[4] Stdev. on third channel = 9.99206377601932
OUT[5] Location of third electrode = [ 2.  0.]
```

InputExtractor subclasses also provide functionality to save the raw data with the specific format for which the InputExtractor was implemented. 

We will now convert our numpy data into the MountainSort format with a MountainSort InputExtractor and our previously defined InputExtractor.

```python
# Write this dataset in the MountainSort format
si.MdaInputExtractor.writeDataset(input_extractor=IX,output_dirname='sample_mountainsort_dataset')
```

InputExtractors can also be used to extract subsets from your data and to extract data from multiple files using SubInputExtractors and MultiInputExtractors, respectively. Examples of these two classes can be seen in the [wiki](https://github.com/colehurwitz31/spikeinterface/wiki). 

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
<br/>

To see our currently implemented subclasses, please check the [extractors](https://github.com/colehurwitz31/spikeinterface/tree/master/spikeinterface/extractors) folder in our repo.

<br/>

## Building a new InputExtractor/OutputExtractor

Building a new InputExtractor or OutputExtractor for specific file format is as simple as creating a new subclass based on the predefined base classes provided in SpikeInterface.

To enable standardization among subclasses, InputExtractor and OutputExtractor are abstract base classes which require a new subclass to override all methods which are decorated with @abstractmethod.

An example of how a new subclass for OutputExtractor can be created is provided below.

```python
from spikeinterface import OutputExtractor

class ExampleOutputExtractor(OutputExtractor):
    def __init__(self, ex_parameter_1, ex_parameter_2):
        OutputExtractor.__init__(self)
        
        ## All file specific initialization code can go here.
        
    def getUnitIds(self):
        
        #Code to get a unit_ids list containing all the ids (ints) of detected units in the recording
        
        return unit_ids

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        
        '''
        Code to get a unit_spike_train 1D array containing all frames (ints) of 
        spikes in the specified unit.
        
        This method will return spike frames within three ranges,
        
                  [start_frame, t_start+1, ..., end_frame-1]
                  [start_frame, start_frame+1, ..., final_unit_spike_frame]
                  [beginning_unit_spike_frame, beginning_unit_spike_frame+1, ..., end_frame-1]
                  
        if both start_frame and end_frame are inputted, if only start_frame is
        inputted, or if only end_frame is inputted, respectively.
        '''
        
        return unit_spike_train
```

As you can see, our extractor base classes were designed to make implementing a new subclass as straightforward and flexible as possible while still enforcing standardized data retrieval functions.

Once all abstract methods are overwritten in your InputExtractor or OutputExtractor, your subclass is ready for deployment and can be used with a variety of pre-implemented tools (links to current tools are contained in the **Tools that use InputExtractors and OutputExtractors** section of the README).
<br/>

## Tools that use InputExtractors and OutputExtractors

Coming soon...
<br/>
<br/>

### Future Plans

Coming soon...
<br/>
<br/>

### Authors

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland 

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Biology (CCB), Flatiron Institute, New York, United States

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Department of Informatics, University of Oslo, Oslo, Norway

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland
<br/>
<br/>
For any correspondence, contact Cole Hurwitz at colehurwitz@gmail.com
