"""Creates counterbalanced run order for hyperface. Assuming that there are
two sessions, and a total of 12 runs, the runs are split into two sessions of
six runs. The sessions are counterbalanced across subjects, and the runs
are randomized within each session. In this way it's possible to reconstruct
consistent sessions across subjects."""
import argparse
import csv
import datetime
from os.path import exists as pexists, join as pjoin
import json
import numpy as np
from random import shuffle
import logging
logging.basicConfig()
logger = logging.getLogger(__file__.replace('.py', ''))
logger.setLevel(logging.INFO)

RUNORDER_ORIG = 'cfg/original_order_runs.json'
RUNORDER_LOGFN = 'cfg/runorder.tsv'
FIELDNAMES_LOG = ['subject_id', 'subject_nr', 'timestamp']


def check_existing(subid):
    """Check if we have already created the runorder for this subject.
    Returns True if `subid` is already in the log"""
    if pexists(RUNORDER_LOGFN):
        with open(RUNORDER_LOGFN, 'rb') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for r in reader:
                if subid.lower() == r['subject_id']:
                    return True
    return False


def get_last_subnr():
    """Returns the last subject number used for any subject in the log"""
    if pexists(RUNORDER_LOGFN):
        with open(RUNORDER_LOGFN, 'rb') as f:
            reader = csv.DictReader(f, delimiter='\t')
            subnrs = [int(r['subject_nr']) for r in reader]
        return max(subnrs)
    else:
        return 0


def get_run_sessions(subnr):
    # load original order
    with open(RUNORDER_ORIG, 'rb') as f:
        order = json.load(f)
    # counterbalance the sessions
    if subnr % 2 == 0:
        session1 = range(6)
        session2 = range(6, 12)
    else:
        session1 = range(6, 12)
        session2 = range(6)
    # shuffle for randomization
    shuffle(session1)
    shuffle(session2)
    # now store everything into a dictionary with labels 'ses-X_run-Y'
    subject_order = dict()
    lbl_template = 'ses-{0}_run-{1}'
    for i_s, session in enumerate([session1, session2], 1):
        for irun, runidx in enumerate(session):
            subject_order[lbl_template.format(i_s, irun+1)] = \
                order[str(runidx)]
    return subject_order


def write_log(subid, subnr):
    writeheader = True if not pexists(RUNORDER_LOGFN) else False
    with open(RUNORDER_LOGFN, 'ab') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES_LOG,
                                delimiter='\t')
        if writeheader:
            writer.writeheader()
        tosave = {
            'subject_id': subid,
            'subject_nr': subnr,
            'timestamp': datetime.datetime.now().isoformat()
        }
        writer.writerow(tosave)



def main():
    parsed = parse_args()
    subid = parsed.subid
    if check_existing(subid):
        logger.warning("Run order for {0} already exists; "
                       "won't recreate it".format(subid))
        return 0
    # get the subject number for this subject
    subnr = get_last_subnr() + 1
    # get this subject order
    subject_order = get_run_sessions(subnr)
    # save it
    subject_order_fn = 'sub-{0}_order_runs.json'.format(subid)
    with open(pjoin('cfg', subject_order_fn), 'wb') as f:
        json.dump(subject_order, f, indent=True, sort_keys=True)
    logging.info("Created {0}".format(subject_order_fn))
    # store it in the log
    write_log(subid, subnr)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--subid', '-s', type=str,
                        help='subjectid for which to create the run order',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.exit(main())