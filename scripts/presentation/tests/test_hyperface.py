from ..hyperface import CFGDIR, SYNCKEY, RESPONSEKEYS, get_order_stimuli, \
    setup_fmri, show_single_movie, SerialEvent
import os
from os.path import join as pjoin, exists as pexists
from os import remove
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
    assert_equal
import numpy as np
import pickle
from psychopy import visual, event
from mock import patch, MagicMock


def test_setup_fmri_test():
    win = visual.Window(fullscr=False, autoLog=False)
    TR = 1
    volumes = 3
    setup_fmri(win, volumes, TR, mode='Test')
    # need to import after call to get global variables
    from ..hyperface import fmri_volumes, fmri_clock

    stored_keys = []
    stored_times = []
    while fmri_clock.getTime() < TR * volumes:
        keys = event.getKeys()
        if keys:
            stored_keys.append(keys)
            stored_times.append(fmri_clock.getTime())
    assert(len(stored_keys) == volumes - 1)  # XXX: we aren't TRlocking
    assert_array_almost_equal(stored_times, np.arange(1, volumes) * TR,
                              decimal=2) # we don't have more than ms
                                         #  precision anyway
    assert_equal(fmri_volumes, 1)


@patch('psychopy.event.getKeys')
@patch('presentation.hyperface.fmri_volumes', 0)
@patch('presentation.hyperface.RESPONSE_TIMESTAMPS', [])
def test_simultaneous_events_show_movie(mocked_getkeys):

    # set up a side effect for getKeys
    class side_effect(object):
        def __init__(self):
            self.count = 0
        def __call__(self, *args, **kwargs):
            if self.count == 0:
                self.count += 1
                return [(SYNCKEY, 0)]
            elif self.count == 1:
                self.count += 1
                return [(SYNCKEY, 3.45), (RESPONSEKEYS[0], 3.46)]
            elif self.count == 2:
                self.count += 1
                return [(RESPONSEKEYS[1], 3.56)]
            else:
                self.count += 1
                return []

    se = side_effect()

    win = visual.Window(fullscr=False, autoLog=False)
    # find where we are
    here = os.path.dirname(os.path.abspath(__file__))
    mov = visual.MovieStim3(win, pjoin(here, 'face001.mp4'), noAudio=True)
    mocked_getkeys.side_effect = se
    show_single_movie(win, mov)
    # load global vars after call to get them updated
    from ..hyperface import fmri_volumes, RESPONSE_TIMESTAMPS, TR

    assert_equal(fmri_volumes, 3)
    assert_array_equal(RESPONSE_TIMESTAMPS, [3.46, 3.56])

@patch('psychopy.event.getKeys')
@patch('serial.Serial')
def test_SerialEvent(mock_serial, mock_getkeys):
    # first of all check that if we do not use any of the *KEY we get the same
    # result as event.getKeys()

    class ReadSideEffect(object):
        def __init__(self):
            self.count = 0

        def __call__(self):
            if self.count == 0:
                self.count += 1
                return SYNCKEY
            elif self.count == 1:
                self.count += 1
                return RESPONSEKEYS[0]
            else:
                self.count += 1
                return ''

    sideeffect = ReadSideEffect()

    se = SerialEvent()
    # I need to mock the MagicMock stored in se.ser now!
    se.ser.read()  # create it
    se.ser.read.side_effect = sideeffect

    se.getKeys(keyList=['b', 'c'])
    # by default we return timestamps internally
    mock_getkeys.assert_called_with(keyList=['b', 'c'], timeStamped=True)

    # check it returns the keys we want
    mock_getkeys.return_value = [('b', 2)]
    allkeys = se.getKeys(keyList=[SYNCKEY, 'b'] + RESPONSEKEYS, timeStamped=True)
    assert_equal(sideeffect.count, 1)
    keys, timestamps = zip(*allkeys)
    assert(SYNCKEY in keys)
    assert('b' in keys)

    allkeys = se.getKeys(keyList=[SYNCKEY, 'b'] + RESPONSEKEYS, timeStamped=True)
    assert_equal(sideeffect.count, 2)
    keys, timestamps = zip(*allkeys)
    assert(RESPONSEKEYS[0] in keys)
    assert('b' in keys)

