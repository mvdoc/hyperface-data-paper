#!/usr/bin/env python
"""Meta script to run the entire experiment"""
import subprocess as sp
from os.path import abspath

DEBUG = False
FULLSCREEN = True
EXIT = 'q'
YES = 'c'
REPEAT = 'r'
SKIP = 's'
FACEMOVIEPY = abspath('./hyperface.py')
facemovie_cmd = FACEMOVIEPY + ' --run_nr {runnr} --ses_nr {sesnr} -s {subid}'

LOCALIZERPY = abspath('../../pitcher_localizer/run_localizer.py')
localizer_cmd = LOCALIZERPY + ' --runnr {runnr} -s {subid}'

def run_facemovie(subid, sesnr, runnr, debug=False, fullscreen=True):
    cmd = facemovie_cmd.format(subid=subid, sesnr=sesnr, runnr=runnr)
    if debug:
        cmd += ' --debug'
    if not fullscreen:
        cmd += ' --no-fullscreen'
    print("+ {0}".format(cmd))
    return sp.check_call(cmd.split())


def run_localizer(subid, sesnr, runnr, debug=False, fullscreen=True):
    runnr_ = runnr + 2 * (sesnr - 1)
    cmd = localizer_cmd.format(subid=subid, sesnr=sesnr, runnr=runnr_)
    if debug:
        cmd += ' --no-scanner'
    if not fullscreen:
        cmd += ' --no-fullscreen'
    print("+ {0}".format(cmd))
    return sp.check_call(cmd.split())


def ask_user(runnr, what):
    txt = "Next run: {0:02d}, {1}. Type {2} to continue, {3} to skip, {4} to repeat " \
          "the previous run: ".format(runnr, what, YES, SKIP, REPEAT)
    out = ''
    while out not in [YES, REPEAT, SKIP]:
        out = raw_input(txt)
    return out

print("#### HYPERFACE EXPERIMENT ####")

subid = ''
while len(subid) != 9:
    subid = raw_input("Enter subid (eg., sid000021): ")
    if len(subid) != 9:
        print("  check subid, {0} doesn't seem correct".format(subid))

sesnr = 0
while sesnr not in [1, 2]:
    sesnr = int(raw_input("Enter session number (1 | 2): "))

runnr = 1
facemovie_runnr = 1
localizer_runnr = 1


while True:
    if runnr in [1, 5]:
        ok = ask_user(runnr, 'localizer')
        if ok in [YES, SKIP]:
            if ok == YES:
                run_localizer(subid, sesnr, localizer_runnr, debug=DEBUG,
                              fullscreen=FULLSCREEN)
            runnr += 1
            localizer_runnr += 1
        elif ok == REPEAT:
            if runnr > 1:
                runnr -= 1
            if localizer_runnr > 1:
                localizer_runnr -= 1

    elif runnr <= 8:
        ok = ask_user(runnr, 'face movie')
        if ok in [YES, SKIP]:
            if ok == YES:
                run_facemovie(subid, sesnr, facemovie_runnr, debug=DEBUG,
                              fullscreen=FULLSCREEN)
            runnr += 1
            facemovie_runnr += 1
        elif ok == REPEAT:
            if runnr > 1:
                runnr -= 1
            if facemovie_runnr > 1:
                facemovie_runnr -= 1
    else:
        print("Done with the experiment")
        break

