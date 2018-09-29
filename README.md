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

You can use the InputExtractor to retrieve data and information from the dataset with a variety of standard functions that are predefined in the InputExtractor base class.

```python
print('Num. channels = {}'.format(IX.getNumChannels()))
print('Sampling frequency = {} Hz'.format(IX.getSamplingFrequency()))
print('Num. timepoints = {}'.format(IX.getNumFrames()))
print('Stdev. on third channel = {}'.format(np.std(IX.getRawTraces(channel_ids=2))))
print('Location of third electrode = {}'.format(IX.getChannelInfo(channel_id=2)['location']))
```
```output
Num. channels = 7
Sampling frequency = 30000 Hz
Num. timepoints = 600000
Stdev. on third channel = 9.99206377601932
Location of third electrode = [ 2.  0.]
```

InputExtractor subclasses also provide functionality to save the raw data with the specific format for which the InputExtractor was implemented. 

We will now convert our numpy data into the MountainSort format with a MountainSort InputExtractor and our previously defined InputExtractor.

```python
# Write this dataset in the MountainSort format
si.MdaInputExtractor.writeInput(input_extractor=IX,output_dirname='sample_mountainsort_dataset')
```

The modular design of InputExtractors allow them to be used in a variety of other tasks. For example, InputExtractors can extract subsets of data from a raw data file or can extract data from multiple files with SubInputExtractors and MultiInputExtractors, respectively. Examples of these two classes can be seen in the [wiki](https://github.com/colehurwitz31/spikeinterface/wiki).

**OutputExtractor**

To run our standardized data retrieval functions for your processed extracellular data, import the subclass OutputExtractor coinciding with your specific file format/spike sorter. Then, you can use that subclass of OutputExtractor to extract data and information from your spike sorted data file. We will show the functionality of the OutputExtractor by continuing our previous example. 

First, we will add some random events and then use the NumpyOutputExtractor to extract data about these events. Generally, OutputExtractors would be instantiated with a path to all the files containing information about the spike sorted units, but since this is a self-contained example, we will add the units manually to the extractor.

```python
# Generate some random events
times=np.sort(np.random.uniform(0,num_timepoints,num_events))
labels=np.random.randint(1,num_units+1,size=num_events)
    
# Define the in-memory output extractor
OX=si.NumpyOutputExtractor()
for k in range(1,num_units+1):
    times_k=times[np.where(labels==k)[0]]
    OX.addUnit(unit_id=k,times=times_k)
```
<br/>

Now, we will demonstrate the API for extracting information from the processed data using standardized functions from the OutputExtractor.

```python
print('Unit ids = {}'.format(OX.getUnitIds()))
st=OX.getUnitSpikeTrain(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1=OX.getUnitSpikeTrain(unit_id=1,start_frame=0,end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
```
```output
Unit ids = [1, 2, 3, 4]
Num. events for unit 1 = 234
Num. events for first second of unit 1 = 16
```


Finally, we can write out our output events to the MountainSort format by using the built-in writeOutput method in the MountainSort OutputExtractor subclass.
```python
si.MdaOutputExtractor.writeOutput(output_extractor=OX,firings_out='sample_mountainsort_dataset/firings_true.mda')
```

Now that we have written out our numpy input and output files in the the MountainSort format, we can easily use the MdaInputExtractor and MdaOutputExtractor for our new datasets and the functionality sould be the same.

```python
# Read this dataset with the Mda input extractor
IX2=si.MdaInputExtractor(dataset_directory='sample_mountainsort_dataset')
OX2=si.MdaOutputExtractor(firings_file='sample_mountainsort_dataset/firings_true.mda')

# We should get the same information as above
print('Unit ids = {}'.format(OX.getUnitIds()))
st=OX2.getUnitSpikeTrain(unit_id=1)
print('Num. events for unit 1 = {}'.format(len(st)))
st1=OX2.getUnitSpikeTrain(unit_id=1,start_frame=0,end_frame=30000)
print('Num. events for first second of unit 1 = {}'.format(len(st1)))
```
```output
Unit ids = [1, 2, 3, 4]
Num. events for unit 1 = 234
Num. events for first second of unit 1 = 16
```
OutputExtractors can also extract subsets of data from a processed data file or can extract data from multiple files with SubOutputExtractors and MultiOutputExtractors, respectively. Examples of these two classes can be seen in the [wiki](https://github.com/colehurwitz31/spikeinterface/wiki).

This concludes the basic tutorial about the Input/Output Extractors. To see currently implemented extractor subclasses, please check the [extractors](https://github.com/colehurwitz31/spikeinterface/tree/master/spikeinterface/extractors) folder in our repo. 

We have also implemented a variety of tools which use InputExtractors and OutputExtractors. Links to these tools are contained in the **Tools that use InputExtractors and OutputExtractors** section of the README.

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
        
    @staticmethod
    def writeOutput(self, output_extractor, save_path):
        
        #This function is not abstract so it is optional if you want to override it. It allows other OutputExtractors
        to use your new OutputExtractor to convert their processed data into your file format.

```

As you can see, our extractor base classes were designed to make implementing a new subclass as straightforward and flexible as possible while still enforcing standardized data retrieval functions.

Once all abstract methods are overwritten in your InputExtractor or OutputExtractor, your subclass is ready for deployment and can be used with any pre-implemented tools (see **Tools that use InputExtractors and OutputExtractors**).
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
