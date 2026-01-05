"""Test the script that makes the runorder"""
from mock import patch
from ..make_runorder import get_run_sessions

def test_get_run_sessions():
    # the order should be counterbalanced for even/odd participants
    odd_sub = get_run_sessions(1)
    even_sub = get_run_sessions(2)

    assert len(odd_sub) == len(even_sub)
    assert len(odd_sub) == 12

    # so these two should be equal
    odd_ses1 = [odd_sub['ses-1_run-{0}'.format(i)] for i in range(1, 7)]
    even_ses2 = [even_sub['ses-2_run-{0}'.format(i)] for i in range(1, 7)]
    # sort them according to run order
    odd_ses1 = sorted(odd_ses1, key=lambda x: x['order_orig'])
    even_ses2 = sorted(even_ses2, key=lambda x: x['order_orig'])
    assert odd_ses1 == even_ses2

    # and also these
    odd_ses2 = [odd_sub['ses-2_run-{0}'.format(i)] for i in range(1, 7)]
    even_ses1 = [even_sub['ses-1_run-{0}'.format(i)] for i in range(1, 7)]
    # sort them according to run order
    odd_ses2 = sorted(odd_ses2, key=lambda x: x['order_orig'])
    even_ses1 = sorted(even_ses1, key=lambda x: x['order_orig'])
    assert odd_ses2 == even_ses1

    # and overall they should be equal
    assert (odd_ses2 + odd_ses1) == (even_ses1 + even_ses2)

    # also check that it's not equal upon two runs
    odd_sub1 = get_run_sessions(1)
    odd_sub2 = get_run_sessions(11)
    assert odd_sub1 != odd_sub2
    # but the division is the same
    keys_ses1 = ['ses-1_run-{0}'.format(i) for i in range(1, 7)]
    keys_ses2 = ['ses-2_run-{0}'.format(i) for i in range(1, 7)]
    for keys in [keys_ses1, keys_ses2]:
        assert(sorted(odd_sub1[key]['order_orig'] for key in keys) ==
               sorted(odd_sub2[key]['order_orig'] for key in keys))
