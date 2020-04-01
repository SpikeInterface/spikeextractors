import numpy as np
from spikeextractors.extraction_tools import load_extractor_from_json


def check_recordings_equal(RX1, RX2):
    N = RX1.get_num_frames()
    # get_channel_ids
    assert np.allclose(RX1.get_channel_ids(), RX2.get_channel_ids())
    # get_num_channels
    assert np.allclose(RX1.get_num_channels(), RX2.get_num_channels())
    # get_num_frames
    assert np.allclose(RX1.get_num_frames(), RX2.get_num_frames())
    # get_sampling_frequency
    assert np.allclose(RX1.get_sampling_frequency(), RX2.get_sampling_frequency())
    # get_traces
    assert np.allclose(RX1.get_traces(), RX2.get_traces())
    sf = 0
    ef = N
    ch = [RX1.get_channel_ids()[0], RX1.get_channel_ids()[-1]]
    assert np.allclose(RX1.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef),
                       RX2.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef))
    for f in range(0, RX1.get_num_frames(), 10):
        assert np.isclose(RX1.frame_to_time(f), RX2.frame_to_time(f))
        assert np.isclose(RX1.time_to_frame(RX1.frame_to_time(f)), RX2.time_to_frame(RX2.frame_to_time(f)))
    # get_snippets
    frames = [30, 50, 80]
    snippets1 = RX1.get_snippets(reference_frames=frames, snippet_len=20)
    snippets2 = RX2.get_snippets(reference_frames=frames, snippet_len=(10, 10))
    for ii in range(len(frames)):
        assert np.allclose(snippets1[ii], snippets2[ii])
        
        
def check_recording_return_types(RX):
    channel_ids = RX.get_channel_ids()
    assert (type(RX.get_num_channels()) == int) or (type(RX.get_num_channels()) == np.int64)
    assert (type(RX.get_num_frames()) == int) or (type(RX.get_num_frames()) == np.int64)
    assert (type(RX.get_sampling_frequency()) == float) or (type(RX.get_sampling_frequency()) == np.float64)
    assert type(RX.get_traces(start_frame=0, end_frame=10)) == np.ndarray
    for channel_id in channel_ids:
        assert (type(channel_id) == int) or (type(channel_id) == np.int64)


def check_sorting_return_types(SX):
    unit_ids = SX.get_unit_ids()
    assert (all(isinstance(id, int) or isinstance(id, np.integer) for id in unit_ids))
    for id in unit_ids:
        train = SX.get_unit_spike_train(id)
        # print(train)
        assert (all(isinstance(x, int) or isinstance(x, np.integer) for x in train))


def check_sortings_equal(SX1, SX2):
    # get_unit_ids
    ids1 = np.sort(np.array(SX1.get_unit_ids()))
    ids2 = np.sort(np.array(SX2.get_unit_ids()))
    assert (np.allclose(ids1, ids2))
    for id in ids1:
        train1 = np.sort(SX1.get_unit_spike_train(id))
        train2 = np.sort(SX2.get_unit_spike_train(id))
        assert np.array_equal(train1, train2)


def check_dumping(extractor):
    extractor.dump(file_name='test.json')
    extractor_loaded = load_extractor_from_json('test.json')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)