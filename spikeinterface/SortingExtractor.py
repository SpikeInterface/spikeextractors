from abc import ABC, abstractmethod
import numpy as np

class SortingExtractor(ABC):
    '''A class that contains functions for extracting important information
    from spiked sorted data given a spike sorting software. It is an abstract
    class so all functions with the @abstractmethod tag must be implemented for
    the initialization to work.


    '''
    def __init__(self):
        self._unit_properties = {}

    @abstractmethod
    def getUnitIds(self):
        '''This function returns a list of ids (ints) for each unit in the recording.

        Returns
        ----------
        unit_ids: array_like
            A list or 1D array of the unit ids in recording (ints).
        '''
        pass

    @abstractmethod
    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        '''This function extracts spike frames from the specified unit.
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

        Parameters
        ----------
        unit_id: int
            The id that specifies a unit in the recording.
        start_frame: int
            The frame above which a spike frame is returned  (inclusive).
        end_frame: int
            The frame below which a spike frame is returned  (exclusive).
        Returns
        ----------
        spike_train: numpy.ndarray
            An 1D array containing all the frames for each spike in the
            specified unit given the range of start and end frames.
        '''
        pass

    def setUnitProperty(self, unit_id, property_name, property_data):
        '''This function adds a unit property data set under the given property
        name

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be added
            (or a list of ids if you want to simultaneously set the property
            for multiple units)
        property_name: str
            A property stored by the sorting extractor (pca_features, etc.)
            (or a list of properties if the unit_id was a list)
        property_data
            The data associated with the given property name. Could be many
            formats as specified by the user.
        '''
        if (type(unit_id)==list) or (type(unit_id)==np.ndarray):
            for i,unit in enumerate(unit_id):
                self.setUnitProperty(unit_id=unit,property_name=property_name,property_data=property_data[i])
            return
        if (isinstance(unit_id, int)) or (isinstance(unit_id, np.int64)):
            if(unit_id in self.getUnitIds()):
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id]={}
                if(isinstance(property_name, str)):
                    self._unit_properties[unit_id][property_name] = property_data
                else:
                    raise ValueError("property_name must be a string")
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")

    def addUnitProperty(self, unit_id, property_name, property_data):
        '''This function adds a unit property data set under the given property
        name

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be added
        property_name: str
            A property stored by the sorting extractor (pca_features, etc.)
        property_data
            The data associated with the given property name. Could be many
            formats as specified by the user.
        '''
        print('WARNING: addUnitProperty is deprecated. Use setUnitProperty instead.')
        if (isinstance(unit_id, int)) or (isinstance(unit_id, np.int64)):
            if(unit_id in self.getUnitIds()):
                if(isinstance(property_name, str)):
                    self._unit_properties[unit_id][property_name] = property_data
                else:
                    raise ValueError("property_name must be a string")
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")

    def getUnitProperty(self, unit_id, property_name):
        '''This function rerturns the data stored under the property name given

        Parameters
        ----------
        unit_id: int
            The unit id for which the property will be returned
            (or a list of unit ids)
        property_name: str
            A property stored by the sorting extractor (pca_features, etc.)
            (or a list of properties if unit_id was a list)
        Returns
        ----------
        property_data
            The data associated with the given property name. Could be many
            formats as specified by the user.
        '''
        if (type(unit_id)==list) or (type(unit_id)==np.ndarray):
            return [self.getUnitProperty(unit_id=unit,property_name=property_name) for unit in unit_id]

        if (isinstance(unit_id, int)) or (isinstance(unit_id, np.int64)):
            if(unit_id in self.getUnitIds()):
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id]={}
                if(isinstance(property_name, str)):
                    if(property_name in list(self._unit_properties[unit_id].keys())):
                        return self._unit_properties[unit_id][property_name]
                    else:
                        raise ValueError("This property has not been added to this unit")
                else:
                    raise ValueError("property_name must be a string")
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")

    def getUnitPropertyNames(self, unit_id):
        if (isinstance(unit_id, int)) or (isinstance(unit_id, np.int64)):
            if(unit_id in self.getUnitIds()):
                if unit_id not in self._unit_properties:
                    self._unit_properties[unit_id]={}
                return sorted(self._unit_properties[unit_id].keys())
            else:
                raise ValueError("Non-valid unit_id")
        else:
            raise ValueError("unit_id must be an int")


    @staticmethod
    def writeSorting(self, sorting_extractor, save_path):
        '''This function writes out the spike sorted data file of a given sorting
        extractor to the file format of this current sorting extractor. Allows
        for easy conversion between spike sorting file formats. It is a static
        method so it can be used without instantiating this sorting extractor.

        Parameters
        ----------
        sorting_extractor: SortingExtractor
            A SortingExtractor that can extract information from the sorted data
            file to be converted to the new format.

        save_path: string
            A path to where the converted sorted data will be saved, which may
            either be a file or a folder, depending on the format.
        '''
        raise NotImplementedError("The writeSorting function is not \
                                  implemented for this extractor")
