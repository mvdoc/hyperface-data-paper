#!/usr/bin/env python
"""
Presentation script for HyperFace
mvdoc May 2015
"""
import argparse
from itertools import permutations
from os.path import join as pjoin, exists as pexists, abspath
from os import makedirs
import numpy as np
import json
from psychopy import visual, core, logging, event as pevent
from psychopy import gui
from psychopy.hardware.emulator import launchScan
import datetime
import time as ptime
import serial
import csv
import shutil
import subprocess as sp


STIMDIR = 'stim'
CFGDIR = 'cfg'
RESDIR = 'res'
SUBJDIR = ''
# XXX: these keys need to be checked on the scanner
SYNCKEY = '5'
RESPONSEKEYS = ['1', '2']
RESPONSE2CODE = {'1': 1, '2': 0}  # 1 is yes, 2 is no
RESPONSE_TIMESTAMPS = []
TR = 1 # TR time in seconds
fmri_clock = core.Clock()
fmri_volumes = 0
SKIP_FLIPS = 4  # number of flips to skip to allow syncing with triggers
SKIP_VOLUMES_BEG = 0  # number of volumes to skip at the beginning
template_bids = '{onset:.3f}\t{duration:.3f}\t{stim}'
# default event handler is psychopy.event
event = pevent

STIM_SIZE = (1280, 720)

# load config
def load_config(config_fn='config.json'):
    with open(config_fn, 'rb') as f:
        config = json.load(f)
    return config


def setup_loggers(subj, runnr, sesnr, debug):
    time_template = '%Y%m%dT%H%M%S'
    config = load_config()
    # ---- LOGGING -----
    log_fn = config['log_template'].format(
        subj=subj,
        task_name=config['task_name'],
        runnr=runnr,
        sesnr=sesnr,
        timestamp=ptime.strftime(time_template),
    )
    log_fn = pjoin(SUBJDIR, log_fn)
    logging.LogFile(log_fn, level=logging.INFO)

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    # add a new level name called bids
    # we will use this level to log information that will be saved
    # in the _events.tsv file for this run
    BIDS = 26
    logging.addLevel(BIDS, 'BIDS')
    logging.console.setLevel(level)

    def move_halted_log(fn):
        # flush log
        logging.flush()
        shutil.move(fn, fn.replace('.txt', '__halted.txt'))
        # quit
        core.quit()

    # set up global key for quitting; if that happens, log will be moved to
    # {log_fn}__halted.txt
    event.globalKeys.add(key='q',
                         modifiers=['ctrl'],
                         func=move_halted_log,
                         func_args=[log_fn],
                         name='quit experiment gracefully')


def logbids(msg, t=None, obj=None, BIDS_LEVEL=26):
    """logbids(message)
    logs a BIDS related message
    """
    logging.root.log(msg, level=BIDS_LEVEL, t=t, obj=obj)



class SerialEvent(object):
    """Extends psychopy.event to get input for serial port"""
    def __init__(self):
        """Open serial port"""
        # set up an internal clock
        # self.clock = core.Clock()
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=.0001)
        self.ser.flushInput()

        self.globalKeys = pevent.globalKeys

    def getKeys(self, keyList=None, timeStamped=False):
        """Use same arguments as event.getKeys"""
        keys = []
        serkey = None
        # if it's the trigger
        if SYNCKEY in keyList:
            # XXX: it can only send one key at a time
            serkey = self.ser.read()
            if serkey == SYNCKEY:
                keys.append((serkey, core.getTime()))
                keyList.remove(SYNCKEY)
        # if it's one of the response keys we want to get them from the serial
        for resp in RESPONSEKEYS:
            if resp in keyList:
                serkey = self.ser.read() if not serkey else serkey
                if serkey == resp:
                    keys.append((serkey, core.getTime()))
                    keyList.remove(resp)
        if keyList is None or keyList:
            keys.extend(pevent.getKeys(keyList=keyList,
                                       timeStamped=True))
        # code borrowed from psychopy.event
        if not timeStamped:
            keyNames = [k[0] for k in keys]
            return keyNames
        elif hasattr(timeStamped, 'getLastResetTime'):
            #keys were originally time-stamped with core.monotonicClock._lastResetTime
            #we need to shift that by the difference between it and our custom clock
            timeBaseDiff = timeStamped.getLastResetTime() - core.monotonicClock.getLastResetTime()
            relTuple = [(k[0], k[1]-timeBaseDiff) for k in keys]
            return relTuple
        elif timeStamped:
            return keys
        elif isinstance(timeStamped, (float, int, long)):
            relTuple = [(k[0], k[1] - timeStamped) for k in keys]
            return relTuple

    @staticmethod
    def waitKeys(*args, **kwargs):
        return pevent.waitKeys(*args, **kwargs)

    def close(self):
        self.ser.close()


def write_subjectlog(fn, info):
    fieldnames = ['subject_id', 'run_nr',
                  'session_nr', 'timestamp']
    info_save = {key: info.get(key, '') for key in fieldnames}
    with open(fn, 'ab') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        info_save['timestamp'] = datetime.datetime.now().isoformat()
        writer.writerow(info_save)


# XXX: there should be two sessions
def load_subjectlog(fn):
    lastinfo = {
        'subject_id': 'sidXXXXXX',
        'subject_nr': 1,
        'run_nr': 1,
        'session_nr': 1
    }
    if not pexists(fn):
        with open(fn, 'wb') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=['subject_id',
                                                'run_nr',
                                                'session_nr',
                                                'timestamp'],
                                    delimiter='\t')
            writer.writeheader()
    else:
        with open(fn, 'rb') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = [r for r in reader]
            lastinfo = rows[-1]
    return lastinfo




# Get subject information
def ask_subinfo():
    subject_log = 'subjectlog.tsv'
    lastinfo = load_subjectlog(subject_log)

    info = {key: lastinfo[key] for key in
            ['subject_id', 'session_nr', 'run_nr']}
    info['debug'] = False
    info['fullscr'] = True

    info_dlg = gui.DlgFromDict(dictionary=info,
                               title='HyperFace',
                               order=['subject_id', 'run_nr', 'session_nr',
                                      'debug', 'fullscr'])
    if not info_dlg.OK:
        core.quit()

    info['run_nr'] = int(info['run_nr'])
    # append info to subject_log
    write_subjectlog(subject_log, info)
    return info


def setup_subjectdir(subject_id):
    global SUBJDIR
    SUBJDIR = pjoin(RESDIR, 'sub-{0}'.format(subject_id))
    try:
        makedirs(SUBJDIR)
    except OSError:
        pass

# NB: all the logic here is moved into `make_runorder.py`
# now we're just loading from that
def get_order_stimuli(subid, runnr, sesnr):
    """
    Load the order of stimuli for this subject.
    order is a list of dictionaries with keys ['run', 'catch']
    """

    subject_order = pjoin(CFGDIR, 'sub-{0}_order_runs.json'.format(subid))
    runlbl = 'ses-{0}_run-{1}'.format(sesnr, runnr)

    if not pexists(subject_order):
        cmd = abspath('./make_runorder.py')
        cmd_ = 'python {cmd} -s {subid}'.format(cmd=cmd, subid=subid)
        print("Generating subject order for {0}".format(subid))
        sp.check_call(cmd_.split())

    with open(subject_order, 'r') as f:
        order = json.load(f)

    return order[runlbl]


def display_txt(win, txt, waitsecs=None, waitkeys=None):
    global event
    stim = visual.TextStim(win, txt, units='pix', height=40, wrapWidth=900)
    stim.draw()
    win.flip()
    if waitsecs:
        core.wait(waitsecs)
    elif waitkeys:
        event.waitKeys(keyList=waitkeys)


# Load stimuli for subject
def load_stimuli(subid, runnr, sesnr, win):
    order = get_order_stimuli(subid, runnr, sesnr)
    display_txt(win, 'Loading stimuli...')
    stimuli = []
    for stim in order['run']:
        print("Loading {0}".format(stim))
        stimuli.append(visual.MovieStim3(win, pjoin(STIMDIR, stim), name=stim,
                                         noAudio=True, size=STIM_SIZE))

    catch = []
    for stim in order['catch']:
        print("Loading {0}".format(stim))
        catch.append(visual.MovieStim3(win, pjoin(STIMDIR, stim),
                                       name='catch_' + stim,
                                         noAudio=True, size=STIM_SIZE))
    display_txt(win, 'Done.', waitsecs=1)
    return stimuli, catch


def show_single_movie(win, mov, duration=4.):
    global RESPONSE_TIMESTAMPS, fmri_volumes, event
    fmri_triggers = 0
    fmri_trigger_times = []
    # wait for a trigger
    #while not event.getKeys(keyList=[SYNCKEY]):
    #    pass
    # update total counter of volumes
    fmri_volumes += 1
    #logging.exp('TR {0:3d}'.format(fmri_volumes))
    # log information for bids
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=duration,
                                 stim=mov.name))
    timer = core.CountdownTimer(duration)
    RESPONSE = -1
    while timer.getTime() > 0:
        mov.draw()
        if mov.status != visual.FINISHED:
            win.flip()
        keys_timestamp = event.getKeys(keyList=[SYNCKEY] + RESPONSEKEYS,
                                       timeStamped=fmri_clock)
        if keys_timestamp:
            for key, timestamp in keys_timestamp:
                if SYNCKEY == key:
                    fmri_triggers += 1
                    fmri_trigger_times.append(timestamp)
                    # update total counter of volumes
                    fmri_volumes += 1
                    #logging.exp('TR {0:3d}'.format(fmri_volumes))
                if key in RESPONSEKEYS:
                    RESPONSE_TIMESTAMPS.append(timestamp)
                    logbids(template_bids.format(onset=timestamp,
                                                 duration=0.00,
                                                 stim='button_press_' + key))
                    RESPONSE = RESPONSE2CODE[key]
    stim_end = timer.getTime()
    #event.clearEvents()
    logging.debug('Stimulus took {0}'.format(duration-stim_end))
    logging.debug('Got {0} triggers during presentation at {1}'.format(
            fmri_triggers, fmri_trigger_times))
    return RESPONSE


def show_stimuli(win, stimuli):
    #while not event.getKeys([SYNCKEY]):
    #    pass
    responses = []
    for istim, stim in enumerate(stimuli):
        responses.append(show_single_movie(win, stim))
        if istim % 5 == 0:
            logging.flush()
    win.clearBuffer()
    win.flip()

    logging.debug('Got {0} responses at times {1}'.format(
            len(RESPONSE_TIMESTAMPS),
            ', '.join([str(timestamp) for timestamp in RESPONSE_TIMESTAMPS]))
    )
    return responses


def setup_fmri(win, volumes=0, tr=2, mode='Test'):
    global fmri_clock, fmri_volumes
    mr_settings = {'TR': tr,
                   'sync': SYNCKEY,
                   'skip': SKIP_VOLUMES_BEG,
                   'volumes': volumes,
                   'sound': False}

    if mode == 'Test':  # we can use psychopy routine
        fmri_volumes = launchScan(win, mr_settings, globalClock=fmri_clock,
                                  mode=mode)
    else:  # need to wait ourselves for the number of tr
        win.mouseVisible = False
        msg = visual.TextStim(win, text='Waiting for Scanner...',
                              autoLog=False)
        msg.draw()
        win.flip()

        skipped_tr = 0
        while skipped_tr < mr_settings['skip']:
            while not event.getKeys([SYNCKEY]):
                pass
            logging.exp('Skipped {0} TRs'.format(skipped_tr + 1))
            logging.flush()
            skipped_tr += 1
        # now wait for the next tr, and reset the clock
        while not event.getKeys([SYNCKEY]):
            pass
        fmri_clock.reset()
        #logging.exp('TR {0:3d}'.format(fmri_volumes))
        # clear up text
        win.flip()
        fmri_volumes = 1
    # get the numbers right
    fmri_volumes += mr_settings['skip']


def wait_for_tr(ntrs, timeout_factor=2):
    """Wait for ntrs with a timeout set by timeout_factor, so it will return
    after having received ntrs or if timeout_factor * ntrs have passed"""
    global fmri_volumes, event
    # set up a clock to timeout in case we miss some tr
    timer = core.CountdownTimer(timeout_factor * ntrs * TR)
    tr = 0
    while tr < ntrs:
        while not event.getKeys([SYNCKEY]):
            if timer.getTime() <= 0:
                logging.warning('wait_for_tr timeout: '
                                'received only {0} TRs'.format(tr + 1))
                return
        logging.debug('Waited for {0} TRs'.format(tr + 1))
        logging.flush()
        tr += 1
        # update total counter of volumes
        fmri_volumes += 1
        #logging.exp('TR {0:3d}'.format(fmri_volumes))


def main(subinfo=None):
    global event
    subid = subinfo['subject_id']
    runnr = int(subinfo['run_nr'])
    sesnr = int(subinfo['session_nr'])
    debug = subinfo['debug']
    fullscr = subinfo['fullscr']
    run = subinfo.get('run', None)
    config = load_config()
    instructions = config['instructions']
    # set up serial handler if not debug
    if not debug:
        serial_event = SerialEvent()
        event = serial_event

    setup_subjectdir(subid)
    setup_loggers(subid, runnr, sesnr, debug)

    # setup logbids
    logbids("onset\tduration\tstimulus")

    # set up window
    size = (1280, 1024)
    win = visual.Window(size=size, allowGUI=False,
                        color=(-1, -1, -1), screen=1, fullscr=fullscr)
    stimuli, catch = load_stimuli(subid, runnr, sesnr, win)
    # Show instructions
    if run:
        #waitkeys = []
        waitsecs = 0
    else:
        #waitkeys=['return'] + RESPONSEKEYS
        waitsecs = 3
    display_txt(win, txt=instructions, waitkeys=[], waitsecs=waitsecs)

    if debug:
        #stimuli = stimuli[:50]
        # each stimulus is 4s long, and then we have 30 s of pauses at the
        # beginning, middle, and end
        volumes = int(4/TR * (len(stimuli) + len(catch)) + 30/TR)
        logging.debug("Setting up emulator with TR {0} and {1} volumes".format(
            TR, volumes))
        setup_fmri(win, volumes, tr=TR, mode='Test')
    else:
        setup_fmri(win, tr=TR, mode='Scan')
    # wait for (some) TRs before starting
    # Fixation
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=10.,
                                 stim='fixation'))

    display_txt(win, '+', waitsecs=10)
    # Stimuli
    show_stimuli(win, stimuli)
    # Fixation
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=10.,
                                 stim='fixation'))
    display_txt(win, '+', waitsecs=10)
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=4.,
                                 stim='instruction'))
    display_txt(win, "Press 1 (left button) if you saw the person in this run.\n\n"
                     "Press 2 (right button) if the person wasn't in this run.",
                waitsecs=4)
    # Catch trials
    responses = show_stimuli(win, catch)
    # Fixation
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=6.,
                                 stim='fixation'))
    display_txt(win, '+', waitsecs=6)
    # get responses to provide accuracy
    run_stim = map(lambda x: x.name, stimuli)
    run_catch = map(lambda x: x.name.replace('catch_', ''), catch)
    correct_responses = np.array([True if c in run_stim else False
                                  for c in run_catch])
    accuracy = np.mean(np.array(responses) == correct_responses)
    logbids(template_bids.format(onset=fmri_clock.getTime(),
                                 duration=4.,
                                 stim='accuracy_{0:.0f}'.format(accuracy*100)))
    display_txt(win, 'Your accuracy was {0:.0f}%'.format(accuracy*100),
                waitsecs=4)
    logging.info("Done in {0}".format(fmri_clock.getTime()))
    logging.flush()
    core.quit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', '-s', type=str,
                        help='subject id')
    parser.add_argument('--run_nr', type=int,
                        help='run nr', choices=range(1, 7))
    parser.add_argument('--ses_nr', type=int,
                        help='session nr', choices=range(1, 3))
    parser.add_argument('--debug', '-d', action='store_true',
                        help='run in debug mode (no trigger)')
    parser.add_argument('--no-fullscreen', action='store_false',
                        help='do not run in fullscreen mode')
    parser.add_argument('--run', action='store_true',
                        help='do not wait for a button press')
    return parser.parse_args()


if __name__ == '__main__':
    parsed = parse_args()
    if parsed.subject_id is None:
       info = ask_subinfo()
    else:
        info = {
            'subject_id': parsed.subject_id,
            'run_nr': parsed.run_nr,
            'session_nr': parsed.ses_nr,
            'debug': parsed.debug,
            'fullscr': parsed.no_fullscreen,
            'run': parsed.run
        }
    main(info)
