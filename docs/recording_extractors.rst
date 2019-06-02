
Recording Extractors
~~~~~~~~~~~~~~~~~~~~

In this tutorial, we will go over what RecordingExtractors are and how
they can be used.

.. code:: python

    import numpy as np
    import spikeextractors as se

Here, we define the properties of the in-memory dataset.

.. code:: python

    num_channels=7
    samplerate=30000
    duration=20
    num_timepoints=int(samplerate*duration)

We generate a pure-noise timeseries dataset recorded by a linear probe
geometry

.. code:: python

    timeseries=np.random.normal(0,10,(num_channels,num_timepoints))
    geom=np.zeros((num_channels,2))
    geom[:,0]=range(num_channels)

Define the in-memory recording extractor

.. code:: python

    RX=se.NumpyRecordingExtractor(timeseries=timeseries,geom=geom,samplerate=samplerate)

We can now print properties that the RecordingExtractor retrieves from
the underlying recording.

.. code:: python

    print('Num. channels = {}'.format(len(RX.get_channel_ids())))
    print('Sampling frequency = {} Hz'.format(RX.get_sampling_frequency()))
    print('Num. timepoints = {}'.format(RX.get_num_frames()))
    print('Stdev. on third channel = {}'.format(np.std(RX.get_traces(channel_ids=2))))
    print('Location of third electrode = {}'.format(RX.get_channel_property(channel_id=2, property_name='location')))


.. parsed-literal::

    Num. channels = 7
    Sampling frequency = 30000.0 Hz
    Num. timepoints = 600000
    Stdev. on third channel = 10.005997614707553
    Location of third electrode = [2. 0.]


Write this dataset in the MountainSort format.

.. code:: python

    se.MdaRecordingExtractor.write_recording(recording=RX,save_path='sample_mountainsort_dataset')

Read this dataset with the Mda recording extractor.

.. code:: python

    RX2=se.MdaRecordingExtractor(dataset_directory='sample_mountainsort_dataset')

.. code:: python

    print('Num. channels = {}'.format(len(RX2.get_channel_ids())))
    print('Sampling frequency = {} Hz'.format(RX2.get_sampling_frequency()))
    print('Num. timepoints = {}'.format(RX2.get_num_frames()))
    print('Stdev. on third channel = {}'.format(np.std(RX2.get_traces(channel_ids=2))))
    print('Location of third electrode = {}'.format(RX.get_channel_property(channel_id=2, property_name='location')))


.. parsed-literal::

    Num. channels = 7
    Sampling frequency = 30000.0 Hz
    Num. timepoints = 600000
    Stdev. on third channel = 10.005997657775879
    Location of third electrode = [2. 0.]


Putting Epochs into our recording (Adding a name to a time period in the
recording).

.. code:: python

    RX2.add_epoch(epoch_name='stimulation', start_frame=1000, end_frame=6000)
    RX2.add_epoch(epoch_name='post_stimulation', start_frame=6000, end_frame=10000)
    RX2.add_epoch(epoch_name='pre_stimulation', start_frame=0, end_frame=1000)
    RX2.get_epoch_names()




.. parsed-literal::

    ['pre_stimulation', 'stimulation', 'post_stimulation']



Return a SubRecordingExtractor that is a view to our epoch. Can view
info about it in parent extractor.

.. code:: python

    RX3 = RX2.get_epoch(epoch_name='stimulation')
    epoch_info = RX2.get_epoch_info('stimulation')
    start_frame = epoch_info['start_frame']
    end_frame = epoch_info['end_frame']
    
    print('Epoch Name = stimulation')
    print('Start Frame = {}'.format(start_frame))
    print('End Frame = {}'.format(end_frame))
    print('Mean. on second channel during stimulation = {}'.format(np.mean(RX3.get_traces(channel_ids=1))))
    print('Location of third electrode = {}'.format(RX.get_channel_property(channel_id=2, property_name='location')))


.. parsed-literal::

    Epoch Name = stimulation
    Start Frame = 1000
    End Frame = 6000
    Mean. on second channel during stimulation = -0.04255314916372299
    Location of third electrode = [2. 0.]


Can extract an arbitrary subset of your data/channels manually without
epoch functionality.

.. code:: python

    RX4=se.SubRecordingExtractor(parent_recording=RX2,channel_ids=[2,3,4,5],start_frame=14000,
                                 end_frame=16000)

Show the information for this sub-dataset.

.. code:: python

    print('Num. channels = {}'.format(len(RX4.get_channel_ids())))
    print('Sampling frequency = {} Hz'.format(RX4.get_sampling_frequency()))
    print('Num. timepoints = {}'.format(RX4.get_num_frames()))
    print('Stdev. on third channel = {}'.format(np.std(RX4.get_traces(channel_ids=2))))
    print('Location of third electrode = {}'.format(RX4.get_channel_property(channel_id=2, property_name='location')))


.. parsed-literal::

    Num. channels = 4
    Sampling frequency = 30000.0 Hz
    Num. timepoints = 2000
    Stdev. on third channel = 10.016402244567871
    Location of third electrode = [2. 0.]


Can rename the sub-dataset channel ids (Channel id mapping: 0–>2, 1–>3,
2–>4, 3–>5).

.. code:: python

    RX5=se.SubRecordingExtractor(parent_recording=RX2,channel_ids=[2,3,4,5], 
                                 renamed_channel_ids = [0,1,2,3],
                                 start_frame=14000,end_frame=16000)
    print('New ids = {}'.format(RX5.get_channel_ids()))
    print('Original ids = {}'.format(RX5.get_original_channel_ids([0,1,2,3])))


.. parsed-literal::

    New ids = [0, 1, 2, 3]
    Original ids = [2, 3, 4, 5]


.. code:: python

    print('Num. channels = {}'.format(len(RX5.get_channel_ids())))
    print('Sampling frequency = {} Hz'.format(RX5.get_sampling_frequency()))
    print('Num. timepoints = {}'.format(RX5.get_num_frames()))
    print('Stdev. on third channel = {}'.format(np.std(RX5.get_traces(channel_ids=0))))
    print('Location of third electrode = {}'.format(RX5.get_channel_property(channel_id=0, property_name='location')))


.. parsed-literal::

    Num. channels = 4
    Sampling frequency = 30000.0 Hz
    Num. timepoints = 2000
    Stdev. on third channel = 10.016402244567871
    Location of third electrode = [2. 0.]

