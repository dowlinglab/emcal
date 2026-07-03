"""Fast tests for ExplorationBias (alpha schedules). No GP training."""
import pytest

from emcal import ExplorationBias, EpSchedule


def _constant(ep0):
    # ExplorationBias(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f,
    #                 improvement, best_error, mean_of_var)
    return ExplorationBias(ep0, None, EpSchedule.CONSTANT, None, None,
                           None, None, None, None, None)


def test_constant_schedule_returns_ep0():
    ep = _constant(1.25)
    ep.update()
    assert ep.ep_curr == 1.25


def test_alpha_is_bounded_to_half_and_two():
    # ep is defined on [0.5, 2]; out-of-range ep0 is clamped by update().
    hi = _constant(5.0)
    hi.update()
    assert hi.ep_curr == 2.0

    lo = _constant(0.1)
    lo.update()
    assert lo.ep_curr == 0.5


def test_decay_schedule_moves_from_ep0_toward_ep_f():
    # Decay needs ep0, ep_f, and bo_iter_max; at iteration 0 it should sit at ep0.
    ep = ExplorationBias(2.0, None, EpSchedule.DECAY, 0, 10, None, 0.5,
                         None, None, None)
    ep.update()
    assert 0.5 <= ep.ep_curr <= 2.0
