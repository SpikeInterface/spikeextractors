
Sorting Extractors
~~~~~~~~~~~~~~~~~~

SortingExtractors are python objects that can extract data and information from sorted data file formats in a standardized and straightfoward way. In the tutorial below, we will go over what SortingExtractors are and how they can be used.they can be used.

.. code:: python

    import numpy as np
    import spikeextractors as se

Here, we define the properties of the in-memory dataset.

.. code:: python

    num_channels=7
    samplerate=30000
    duration=20
    num_timepoints=int(samplerate*duration)
    num_units=4
    num_events=1000

We generate a pure-noise timeseries dataset recorded by a linear probe
geometry, generate some random events, and then define in-memory sorting
and recording extractors.

.. code:: python

    # Generate a pure-noise timeseries dataset and a linear geometry
    timeseries=np.random.normal(0,10,(num_channels,num_timepoints))
    geom=np.zeros((num_channels,2))
    geom[:,0]=range(num_channels)
    
    # Define the in-memory recording extractor
    RX=se.NumpyRecordingExtractor(timeseries=timeseries,geom=geom,samplerate=samplerate)
    
    # Generate some random events
    times=np.int_(np.sort(np.random.uniform(0,num_timepoints,num_events)))
    labels=np.random.randint(1,num_units+1,size=num_events)
        
    # Define the in-memory sorting extractor
    SX=se.NumpySortingExtractor()
    for k in range(1,num_units+1):
        times_k=times[np.where(labels==k)[0]]
        SX.add_unit(unit_id=k,times=times_k)

We can now print properties that the SortingExtractor retrieves from the
underlying sorted dataset.

.. code:: python

    print('Unit ids = {}'.format(SX.get_unit_ids()))
    st=SX.get_unit_spike_train(unit_id=1)
    print('Num. events for unit 1 = {}'.format(len(st)))
    st1=SX.get_unit_spike_train(unit_id=1,start_frame=0,end_frame=30000)
    print('Num. events for first second of unit 1 = {}'.format(len(st1)))


.. parsed-literal::

    Unit ids = [1, 2, 3, 4]
    Num. events for unit 1 = 273
    Num. events for first second of unit 1 = 16


Write the recording and sorting in the MountainSort format.

.. code:: python

    se.MdaRecordingExtractor.write_recording(recording=RX,save_path='sample_mountainsort_dataset')
    se.MdaSortingExtractor.write_sorting(sorting=SX,save_path='sample_mountainsort_dataset/firings_true.mda')

Read these new datasets with the Mda recording and sorting extractor.

.. code:: python

    RX2=se.MdaRecordingExtractor(dataset_directory='sample_mountainsort_dataset')
    SX2=se.MdaSortingExtractor(firings_file='sample_mountainsort_dataset/firings_true.mda')

We should get he same information as above.

.. code:: python

    print('Unit ids = {}'.format(SX2.get_unit_ids()))
    st=SX2.get_unit_spike_train(unit_id=1)
    print('Num. events for unit 1 = {}'.format(len(st)))
    st1=SX2.get_unit_spike_train(unit_id=1,start_frame=0,end_frame=30000)
    print('Num. events for first second of unit 1 = {}'.format(len(st1)))


.. parsed-literal::

    Unit ids = [1 2 3 4]
    Num. events for unit 1 = 273
    Num. events for first second of unit 1 = 16


Unit properties are name value pairs that we can store for any unit. We
will now calculate a unit property and store it in the SortingExtractor.

.. code:: python

    full_spike_train = SX2.get_unit_spike_train(unit_id=1)
    firing_rate = float(len(full_spike_train))/RX2.get_num_frames()
    SX2.set_unit_property(unit_id=1, property_name='firing_rate', value=firing_rate)
    print('Average firing rate during the recording of unit 1 = {}'.format(SX2.get_unit_property(unit_id=1, property_name='firing_rate')))
    print("Spike property names: " + str(SX2.get_unit_property_names()))


.. parsed-literal::

    Average firing rate during the recording of unit 1 = 0.000455
    Spike property names: ['firing_rate']


We can get a the sub-dataset from the sorting.

.. code:: python

    SX3=se.SubSortingExtractor(parent_sorting=SX2,unit_ids=[1, 2],
                               start_frame=10000,end_frame=20000)

.. code:: python

    print('Num. units = {}'.format(len(SX3.get_unit_ids())))
    print('Average firing rate of units1 during frames 14000-16000 = {}'.format(
                            float(len(SX3.get_unit_spike_train(unit_id=1)))/6000))


.. parsed-literal::

    Num. units = 2
    Average firing rate of units1 during frames 14000-16000 = 0.001


We can add features to spikes contained in any unit as shown below

.. code:: python

    SX3.set_unit_spike_features(unit_id=1, feature_name='amplitude',
                                value=[55, 60, 64, 50, 54, 60])
    print("Spike feature names: " + str(SX3.get_unit_spike_feature_names()))


.. parsed-literal::

    Spike feature names: ['amplitude']

