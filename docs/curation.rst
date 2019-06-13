
Curation Tutorial
~~~~~~~~~~~~~~~~~

In this tutorial, we will go over what CurationSortingExtractors are and
how they can be used to curate the results in a SortingExtractor.

.. code:: python

    import numpy as np
    import spikeextractors as se

Here, we define the properties of the in-memory dataset.

.. code:: python

    num_channels=7
    samplerate=30000
    duration=20
    num_timepoints=int(samplerate*duration)
    num_units=5
    num_events=20

We generate a pure-noise timeseries dataset recorded by a linear probe
geometry, generate some random events, and then define in-memory sorting
and recording extractors.

We will add some spike features to the units to show how splitting and
merging units effect spike features. Any spike features that are shared
across *all* units will be split and merged correctly, otherwise the
features that are not shared across split and merged units will be
removed from the CurationSortingExtractor.

Unit properties will automatically be removed from the
CurationSortingExtractor when splitting and mergin as they are not
well-defined for those operations.

.. code:: python

    # Generate a pure-noise timeseries dataset and a linear geometry
    np.random.seed(0)
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
        
    #Add some features to the sorting extractor. These will be merged and split appropriately during curation
    spikes = 0
    for unit_id in SX.get_unit_ids():
        SX.set_unit_spike_features(unit_id, feature_name='f_int', value=range(spikes, spikes + len(SX.get_unit_spike_train(unit_id))))
        spikes += len(SX.get_unit_spike_train(unit_id))
        
    spikes = 0
    for unit_id in SX.get_unit_ids():
        SX.set_unit_spike_features(unit_id, feature_name='f_float', value=np.arange(float(spikes) + .1, float(spikes + len(SX.get_unit_spike_train(unit_id) + .1))))
        spikes += len(SX.get_unit_spike_train(unit_id))
        
    #Features that are not shared across ALL units will not be merged and split correctly (will disappear)
    SX.set_unit_spike_features(1, feature_name='bad_feature', value=np.repeat(1, len(SX.get_unit_spike_train(1))))
    SX.set_unit_spike_features(2, feature_name='bad_feature', value=np.repeat(2, len(SX.get_unit_spike_train(2))))
    SX.set_unit_spike_features(3, feature_name='bad_feature', value=np.repeat(3, len(SX.get_unit_spike_train(3))))

.. code:: python

    print('Unit ids = {}'.format(SX.get_unit_ids()))
    st=SX.get_unit_spike_train(unit_id=1)
    print('Num. events for unit 1 = {}'.format(len(st)))


.. parsed-literal::

    Unit ids = [1, 2, 3, 4, 5]
    Num. events for unit 1 = 5


Now we can curate the results using a CurationSortingExtractor.

.. code:: python

    CSX = se.CurationSortingExtractor(parent_sorting=SX)

.. code:: python

    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Original Unit Ids: " + str(SX.get_unit_ids()))
    
    print("Curated ST: " + str(CSX.get_unit_spike_train(1)))
    print("Original ST: " + str(SX.get_unit_spike_train(1)))


.. parsed-literal::

    Curated Unit Ids: [1, 2, 3, 4, 5]
    Original Unit Ids: [1, 2, 3, 4, 5]
    Curated ST: [206907 220517 331138 430220 574290]
    Original ST: [206907 220517 331138 430220 574290]


Lets split one unit from the sorting result (this could be two units
incorrectly clustered as one)

.. code:: python

    CSX.split_unit(unit_id=1, indices=[0, 1])
    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Original Spike Train: " + str(SX.get_unit_spike_train(1)))
    print("Split Spike Train 1: " + str(CSX.get_unit_spike_train(6)))
    print("Split Spike Train 2: " + str(CSX.get_unit_spike_train(7)))
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Unit Ids: [2, 3, 4, 5, 6, 7]
    Original Spike Train: [206907 220517 331138 430220 574290]
    Split Spike Train 1: [206907 220517]
    Split Spike Train 2: [331138 430220 574290]
    2
    
    3
    
    4
    
    5
    
    6
    ^-------1
    
    7
    ^-------1
    


If the split was incorrect, we can always merge the two units back
together.

.. code:: python

    CSX.merge_units(unit_ids=[6, 7])
    print("Curated Spike Train: " + str(CSX.get_unit_spike_train(8)))
    print("Original Spike Train: " + str(SX.get_unit_spike_train(1)))
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Spike Train: [206907 220517 331138 430220 574290]
    Original Spike Train: [206907 220517 331138 430220 574290]
    2
    
    3
    
    4
    
    5
    
    8
    ^-------6
    	^-------1
    ^-------7
    	^-------1
    


We can also exclude units, so let’s get rid of 8 since we are seem to be
confused about this unit.

.. code:: python

    CSX.exclude_units(unit_ids=[8])
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    2
    
    3
    
    4
    
    5
    


Now let’s merge 3 and 4 together (This will create a new unit which
encapsulates both previous units).

.. code:: python

    CSX.merge_units(unit_ids=[3, 4])
    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Merged Spike Train: " + str(CSX.get_unit_spike_train(9)))
    print("Original Spike Trains concatenated: " + str(np.sort(np.concatenate((SX.get_unit_spike_train(3), SX.get_unit_spike_train(4))))))
    print("\nCuration Tree")
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Unit Ids: [2, 5, 9]
    Merged Spike Train: [183155 210132 220886 398518 445947 477836 507142]
    Original Spike Trains concatenated: [183155 210132 220886 398518 445947 477836 507142]
    
    Curation Tree
    2
    
    5
    
    9
    ^-------3
    ^-------4
    


Now let’s merge units 2 and 6 together.

.. code:: python

    CSX.merge_units(unit_ids=[2, 9])
    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Merged Spike Train: " + str(CSX.get_unit_spike_train(10)))
    merged_spike_train = []
    for unit_id in SX.get_unit_ids():
        if(unit_id != 1 and unit_id != 5):
            merged_spike_train.append(SX.get_unit_spike_train(unit_id))
    merged_spike_train = np.asarray(merged_spike_train)
    merged_spike_train = np.sort(np.concatenate(merged_spike_train).ravel())
    print("Original Spike Trains concatenated: " + str(merged_spike_train))
    print("\nCuration Tree")
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Unit Ids: [5, 10]
    Merged Spike Train: [183155 210132 220886 327869 398518 436875 445947 477836 507142 525257]
    Original Spike Trains concatenated: [183155 210132 220886 327869 398518 436875 445947 477836 507142 525257]
    
    Curation Tree
    5
    
    10
    ^-------2
    ^-------9
    	^-------3
    	^-------4
    


Now let’s split unit 5 with given indices.

.. code:: python

    CSX.split_unit(unit_id=5, indices=[0, 1])
    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Original Spike Train: " + str(SX.get_unit_spike_train(5)))
    print("Split Spike Train 1: " + str(CSX.get_unit_spike_train(11)))
    print("Split Spike Train 2: " + str(CSX.get_unit_spike_train(12)))
    print("\nCuration Tree")
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Unit Ids: [10, 11, 12]
    Original Spike Train: [168716 256926 272397 318528 470153]
    Split Spike Train 1: [168716 256926]
    Split Spike Train 2: [272397 318528 470153]
    
    Curation Tree
    10
    ^-------2
    ^-------9
    	^-------3
    	^-------4
    
    11
    ^-------5
    
    12
    ^-------5
    


Finally, we can merge units 10 and 11.

.. code:: python

    CSX.merge_units(unit_ids=[10, 11])
    print("Curated Unit Ids: " + str(CSX.get_unit_ids()))
    print("Merged Spike Train: " + str(CSX.get_unit_spike_train(13)))
    original_spike_train = (np.sort(np.concatenate((SX.get_unit_spike_train(3), SX.get_unit_spike_train(4), SX.get_unit_spike_train(2), SX.get_unit_spike_train(5)[np.asarray([0,1])]))))
    print("Original Spike Train: " + str(original_spike_train))
    print("\nCuration Tree")
    for unit_id in CSX.get_unit_ids():
        CSX.printCurationTree(unit_id=unit_id)


.. parsed-literal::

    Curated Unit Ids: [12, 13]
    Merged Spike Train: [168716 183155 210132 220886 256926 327869 398518 436875 445947 477836
     507142 525257]
    Original Spike Train: [168716 183155 210132 220886 256926 327869 398518 436875 445947 477836
     507142 525257]
    
    Curation Tree
    12
    ^-------5
    
    13
    ^-------10
    	^-------2
    	^-------9
    		^-------3
    		^-------4
    ^-------11
    	^-------5
    


We will now write the input/output in the MountainSort format.

.. code:: python

    se.MdaRecordingExtractor.write_recording(recording=RX,save_path='sample_mountainsort_dataset')
    se.MdaSortingExtractor.write_sorting(sorting=CSX,save_path='sample_mountainsort_dataset/firings_true.mda')

We can read this dataset with the Mda input extractor (we can now have a
normal sorting extractor with our curations).

.. code:: python

    RX2=se.MdaRecordingExtractor(dataset_directory='sample_mountainsort_dataset')
    SX2=se.MdaSortingExtractor(firings_file='sample_mountainsort_dataset/firings_true.mda')

.. code:: python

    print("New Unit Ids: " + str(SX2.get_unit_ids()))
    print("New Unit Spike Train: " + str(SX2.get_unit_spike_train(13)))
    print("Previous Curated Unit Spike Train: " + str(CSX.get_unit_spike_train(13)))


.. parsed-literal::

    New Unit Ids: [12 13]
    New Unit Spike Train: [168716 183155 210132 220886 256926 327869 398518 436875 445947 477836
     507142 525257]
    Previous Curated Unit Spike Train: [168716 183155 210132 220886 256926 327869 398518 436875 445947 477836
     507142 525257]

