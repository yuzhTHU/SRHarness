"""时延表达式的公开行为示例。"""

import numpy as np

import sr_harness_engine as engine


def test_delay_interpolates_along_the_leading_time_axis():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    data = {
        "x": np.array([0.0, 1.0, 4.0, 9.0]),
        "delta": np.full(4, 0.5),
    }
    delayed = engine.parse("delay(x, delta)").evaluate(data, time=time)

    assert np.isnan(delayed[0])
    assert np.allclose(delayed[1:], [0.5, 2.5, 6.5])


def test_delay_applies_to_every_feature_column():
    data = {
        "x": np.array([[0.0, 10.0], [1.0, 20.0], [4.0, 40.0]]),
        "delta": np.ones(3),
    }
    delayed = engine.parse("delay(x, delta)").evaluate(data)

    assert np.isnan(delayed[0]).all()
    assert np.array_equal(delayed[1:], [[0.0, 10.0], [1.0, 20.0]])


def test_an_ode_evaluator_can_supply_its_own_history_lookup():
    calls = []

    def history_lookup(value, lag, time):
        calls.append((value.copy(), lag.copy(), time.copy()))
        return np.full_like(value, 42.0)

    data = {"x": np.arange(3.0), "delta": np.ones(3)}
    result = engine.parse("x + delay(x, delta)").evaluate(
        data,
        time=np.arange(3.0),
        delay_resolver=history_lookup,
    )

    assert np.array_equal(result, [42.0, 43.0, 44.0])
    assert len(calls) == 1
