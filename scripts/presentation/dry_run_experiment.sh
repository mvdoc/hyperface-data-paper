#!/bin/bash -e

for ses in `seq 1 2`; do
	for run in `seq 1 6`; do
		echo "Running session $ses, run $run"
		python hyperface.py -s test --run_nr $run --ses_nr $ses -d --run
	done
done
