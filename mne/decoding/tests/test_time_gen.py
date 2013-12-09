# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true

from mne import fiff, Epochs, read_events
from mne.utils import _TempDir, requires_sklearn
from mne.decoding import time_generalization

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)


@requires_sklearn
def test_time_generalization():
    """Test time generalization decoding
    """
    raw = fiff.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = fiff.pick_types(raw.info, meg='mag', stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True, decim=decim)

    epochs_list = [epochs[k] for k in event_id.keys()]
    scores = time_generalization(epochs_list, cv=2, random_state=42)
    n_times = len(epochs.times)
    assert_true(scores.shape == (n_times, n_times))
    assert_true(scores.max() <= 1.)
    assert_true(scores.min() >= 0.)