# HyperFace
This folder contains the presentation code for HyperFace.

The experiment is divided into two sessions, each session is divided into 6 runs. Within each run, participants watch a continuous stream of 4s clips obtained from YouTube. Then, they are asked to report whether four clips were presented in the run, as an attention check.

Randomization occurs at the level of runs within each session. The order of the stimuli is maintained constant in order to recover the same response across participants, but the run order is randomized for each participant. In addition, runs are randomized only within each session, and the order is counterbalanced across participants. This means that given two subjects, `sub-01` and `sub-02`, they might have the following run orders

Subject  | Session 1           | Session 2
---------|---------------------|--------------------
`sub-01` | 1, 3, 6, 5, 4, 2    | 12, 9, 10, 11, 8, 7
`sub-02` | 11, 8, 9, 12, 10, 7 | 5, 4, 6, 3, 1, 2

in this way it's always possible to return to the standard original sequence.


## How To

1. First you need to create the run order for the participant. Simply use the python script `make_runorder.py` and use the participants identifier. The script will generate the run order for this participant and save it under `cfg/sub-id_order_runs.json`, and store the information in `cfg/runorder.tsv`.
2. Run the experiment using `hyperface.py`. Make sure to use the same identifier.
