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


def test_decay_schedule_reaches_ep_f_after_decay_steps():
    # decay_steps = bo_iter_max // 2; at/after that iteration, ep is clamped to ep_f.
    ep = ExplorationBias(2.0, None, EpSchedule.DECAY, 6, 10, None, 0.5,
                         None, None, None)
    ep.update()
    assert ep.ep_curr == 0.5


def _boyle(ep0, ep_curr, ep_inc, improvement):
    # ExplorationBias(ep0, ep_curr, ep_enum, bo_iter, bo_iter_max, ep_inc, ep_f,
    #                 improvement, best_error, mean_of_var)
    return ExplorationBias(ep0, ep_curr, EpSchedule.BOYLE, None, None, ep_inc, None,
                           improvement, None, None)


def test_boyle_schedule_uses_ep0_on_first_call():
    ep = _boyle(ep0=1.5, ep_curr=None, ep_inc=1.2, improvement=None)
    ep.update()
    assert ep.ep_curr == 1.5


def test_boyle_schedule_decreases_exploration_after_improvement():
    ep = _boyle(ep0=1.0, ep_curr=1.2, ep_inc=1.2, improvement=True)
    ep.update()
    assert ep.ep_curr == pytest.approx(1.0)


def test_boyle_schedule_increases_exploration_without_improvement():
    ep = _boyle(ep0=1.0, ep_curr=1.0, ep_inc=1.2, improvement=False)
    ep.update()
    assert ep.ep_curr == pytest.approx(1.2)


def _jasrasaria(best_error, mean_of_var):
    return ExplorationBias(None, None, EpSchedule.JASRASARIA, None, None, None, None,
                           None, best_error, mean_of_var)


def test_jasrasaria_schedule_uses_variance_ratio_when_best_error_positive():
    ep = _jasrasaria(best_error=2.0, mean_of_var=1.0)
    ep.update()
    assert ep.ep_curr == pytest.approx(1 + 1.0 / 2.0**2)


def test_jasrasaria_schedule_falls_back_to_ep_max_when_best_error_non_positive():
    ep = _jasrasaria(best_error=0.0, mean_of_var=1.0)
    ep.update()
    assert ep.ep_curr == 2.0
