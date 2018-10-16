from spikeinterface import SortingExtractor
from spikeinterface.tools import read_python

import numpy as np
import os
from os.path import join
import neo
import h5py

class KlustaSortingExtractor(SortingExtractor):
    def __init__(self, kwikfile, prmfile=None):
        SortingExtractor.__init__(self)
        #TODO use h5py instead
        f_= h5py.File(kwikfile)
        raise Exception()
        if os.path.exists(kwikfile):
            kwikio = neo.io.KwikIO(filename=kwikfile, )
            blk = kwikio.read_block(raw_data_units='uV')
            self._spiketrains = blk.segments[0].spiketrains
            self._unit_ids = [st.annotations['cluster_id'] for st in self._spiketrains]
        else:
            raise Exception('No kwik file!')

        if prmfile is None:
            prmfile = [f for f in os.listdir(os.path.dirname(kwikfile)) if f.endswith('prm')]
            if len(prmfile) == 1:
                params = read_python(join(os.path.dirname(kwikfile),prmfile[0]))
                self._fs = params['traces']['sample_rate']
            elif len(prmfile) == 1:
                raise AttributeError("klusta .prm file not found! Indicate .prm file with 'prmfile argument'")
            else:
                raise AttributeError("Found multiple klusta .prm files! Indicate .prm file with 'prmfile argument'")

        # set unit properties
        for i_s, spiketrain in enumerate(self._spiketrains):
            for key, val in spiketrain.annotations.items():
                self.setUnitProperty(self.getUnitIds()[i_s], key, val)

    def getUnitIds(self):
        return list(self._unit_ids)

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf
        times = self._spiketrains[self.getUnitIds().index(unit_id)].rescale('s').magnitude * self._fs
        inds = np.where((start_frame <= times) & (times < end_frame))
        return times[inds]

    @staticmethod
    def writeSorting(sorting, save_path):
        pass