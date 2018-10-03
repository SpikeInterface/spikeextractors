import spikeinterface as si
import numpy as np

class MultiInputExtractor(si.InputExtractor):
    def __init__(self,input_extractors,epoch_names=None):
        if epoch_names is None:
            epoch_names=[str(i) for i in range(len(input_extractors))]
        self._IXs=input_extractors
        self._epoch_names=epoch_names
        
        #Num channels and sampling frequency based off the initial extractor
        self._first_input_extractor = input_extractors[0]
        self._num_channels = self._first_input_extractor.getNumChannels()
        self._sampling_frequency = self._first_input_extractor.getSamplingFrequency()

        #Test if all input_extractors have same num channels and sampling frequency
        for i, input_extractor in enumerate(input_extractors[1:]):
            num_channels = input_extractor.getNumChannels()
            sampling_frequency = input_extractor.getSamplingFrequency()

            if (self._num_channels != num_channels):
                raise ValueError("Inconsistent number of channels between extractor 0 and extractor " + str(i + 1))
            if (self._sampling_frequency != sampling_frequency):
                raise ValueError("Inconsistent sampling frequency between extractor 0 and extractor " + str(i + 1))
        
        self._start_frames=[]
        ff=0
        for IX in self._IXs:
            self._start_frames.append(ff)
            ff=ff+IX.getNumFrames()
        self._start_frames.append(ff)
        self._num_frames=ff
        
    def _find_section_for_frame(self, frame):
        inds=np.where(np.array(self._start_frames[:-1])<=frame)[0]
        ind=inds[-1]
        return self._IXs[ind], ind, frame-self._start_frames[ind]
            
    def getRawTraces(self, start_frame=None, end_frame=None, channel_ids=None):
        if start_frame is None:
            start_frame=0
        if end_frame is None:
            end_frame=self.getNumFrames()
        IX1, i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        IX2, i_sec2, i_end_frame = self._find_section_for_frame(end_frame)
        if i_sec1==i_sec2:
            return IX1.getRawTraces(start_frame=i_start_frame,end_frame=i_end_frame,channel_ids=channel_ids)
        list=[]
        list.append(
            self._IXs[i_sec1].getRawTraces(start_frame=i_start_frame,end_frame=self._IXs[i_sec1].getNumFrames(),channel_ids=channel_ids)
        )
        if i_sec in range(i_sec1+1,i_sec2):
            list.append(
                self._IXs[i_sec].getRawTraces(channel_ids=channel_ids)
            )
        list.append(
            self._IXs[i_sec2].getRawTraces(start_frame=0,end_frame=i_end_frame,channel_ids=channel_ids)
        )
        return np.concatenate(list,axes=1)

    def getNumChannels(self):
        return self._num_channels

    def getNumFrames(self):
        return self._num_frames

    def getSamplingFrequency(self):
        return self._sampling_frequency

    def frameToTime(self, frame):
        frame=np.array(frame)
        ret=np.zeros(frame.shape)
        min_frame=np.min(frame)
        max_frame=np.max(frame)
        IX1, i_sec1, i_start_frame = self._find_section_for_frame(start_frame)
        IX2, i_sec2, i_end_frame = self._find_section_for_frame(end_frame)
        for i_sec in range(i_sec1,i_sec2+1):
            IX=self._IXs[i_sec]
            inds=np.where(
                (self._start_frames[i_sec]<=frame)&(frame<self._start_frames[i_sec+1])
            )[0]
            ret[inds]=IX.frameToTime(frame-self._start_frames[i_sec])
        return ret

    def timeToFrame(self, time):
        raise NotImplementedError("The timeToFrame function is not \
                                  implemented for this extractor")

    def getChannelInfo(self, channel_id):
        return self._first_input_extractor.getChannelInfo(channel_id)
    
    def getEpochNames(self):
        return self._epoch_names
    
    def getEpochInfo(self,epoch_name):
        ind=self._epoch_names.index(epoch_name)
        return dict(
            start_frame=self._start_frames[ind],
            end_frame=self._start_frames[ind+1]
        )
    