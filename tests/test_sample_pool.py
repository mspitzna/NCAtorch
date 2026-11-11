from unittest.mock import Mock

import pytest
import torch

from nca.training.sample_pool import SamplePool, TimeseriesSamplePool


@pytest.mark.parametrize("mutation_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "damage_ratio, expected_damaged",
    [(0.0, 0), (0.01, 0), (0.1, 1), (0.29, 2), (0.5, 5), (1.0, 10)],
)
def test_damage_ratio_selects_fraction_of_pool_samples(
    mutation_ratio, damage_ratio, expected_damaged
):
    pool = SamplePool(
        seed_ratio=0.5, damage_ratio=damage_ratio, mutation_ratio=mutation_ratio
    )
    pool_data = torch.ones(10, 2, 4, 4)
    pool.commit(pool_data, torch.ones(10, 1), torch.ones(10, 1))
    # Use an unmistakable damage marker, independent of random circle geometry.
    pool.apply_damage = Mock(side_effect=torch.zeros_like)

    data, cond, true = pool.sample_and_replace(
        torch.full((20, 2, 4, 4), 2.0), torch.zeros(20, 1), torch.zeros(20, 1)
    )

    fresh = (data == 2).flatten(1).all(dim=1)
    intact = (data == 1).flatten(1).all(dim=1)
    damaged = (data == 0).flatten(1).all(dim=1)
    assert fresh.sum().item() == 10
    assert intact.sum().item() == 10 - expected_damaged
    assert damaged.sum().item() == expected_damaged
    assert pool.get_damage_ratio() == damage_ratio
    if expected_damaged:
        pool.apply_damage.assert_called_once()
        torch.testing.assert_close(
            pool.apply_damage.call_args.args[0],
            torch.ones(expected_damaged, 2, 4, 4),
        )
    else:
        pool.apply_damage.assert_not_called()

    # Damage must not alter stored states, conditioning, or targets.
    torch.testing.assert_close(torch.stack(pool.data_pool), pool_data)
    torch.testing.assert_close(cond, true)
    assert true.sum().item() == int(10 * (1 - mutation_ratio))
    assert torch.all(true[fresh] == 0)


@pytest.mark.parametrize("seed_ratio, pool_count", [(0.0, 10), (0.5, 0), (0.5, 9)])
def test_no_damage_when_no_pool_samples_are_drawn(seed_ratio, pool_count):
    pool = SamplePool(seed_ratio=seed_ratio, damage_ratio=1.0)
    pool.commit(torch.ones(pool_count, 2, 4, 4), None, torch.ones(pool_count, 1))
    pool.apply_damage = Mock(side_effect=torch.zeros_like)
    fresh = torch.full((20, 2, 4, 4), 2.0)

    data, cond, true = pool.sample_and_replace(fresh.clone(), None, torch.zeros(20, 1))

    torch.testing.assert_close(data, fresh)
    assert cond is None
    torch.testing.assert_close(true, torch.zeros(20, 1))
    pool.apply_damage.assert_not_called()


def test_damage_fraction_respects_damage_delay():
    pool = SamplePool(damage_ratio=0.5)
    pool.enable_seed_scheduling(total_steps=10, start_ratio=0.5, end_ratio=0.5)
    pool.enable_damage_delay(2)
    pool.commit(torch.ones(10, 2, 4, 4), None, torch.ones(10, 1))
    pool.apply_damage = Mock(side_effect=torch.zeros_like)

    for step, expected_damaged in [(1, 0), (2, 5)]:
        pool.step(step)
        data, _, _ = pool.sample_and_replace(
            torch.full((20, 2, 4, 4), 2.0), None, torch.zeros(20, 1)
        )
        assert (data == 0).flatten(1).all(dim=1).sum().item() == expected_damaged

    pool.apply_damage.assert_called_once()


@pytest.mark.parametrize(
    "damage_ratio, expected_damaged",
    [(0.0, 0), (0.01, 0), (0.1, 1), (0.29, 2), (0.5, 5), (1.0, 10)],
)
def test_timeseries_damage_ratio_selects_fraction_of_reused_samples(
    damage_ratio, expected_damaged
):
    pool = TimeseriesSamplePool(seed_ratio=0.5, damage_ratio=damage_ratio)
    previous_cond = torch.stack((torch.arange(20), torch.zeros(20)), dim=1)
    current_cond = previous_cond + torch.tensor([0, 1])
    pool_data = torch.ones(20, 2, 4, 4)
    pool.commit(pool_data, previous_cond, torch.ones(20, 1))
    pool.apply_damage = Mock(side_effect=torch.zeros_like)

    data, cond, true = pool.sample_and_replace(
        torch.full_like(pool_data, 2.0), current_cond, torch.zeros(20, 1)
    )

    assert (data == 2).flatten(1).all(1).sum().item() == 10
    assert (data == 1).flatten(1).all(1).sum().item() == 10 - expected_damaged
    assert (data == 0).flatten(1).all(1).sum().item() == expected_damaged
    if expected_damaged:
        pool.apply_damage.assert_called_once()
        torch.testing.assert_close(
            pool.apply_damage.call_args.args[0],
            torch.ones(expected_damaged, 2, 4, 4),
        )
    else:
        pool.apply_damage.assert_not_called()
    torch.testing.assert_close(cond, current_cond)
    torch.testing.assert_close(true, torch.zeros(20, 1))
    torch.testing.assert_close(torch.stack(pool.data_pool), pool_data)


@pytest.mark.parametrize("damage_ratio, expected_damaged", [(0.1, 0), (0.5, 3), (1.0, 7)])
def test_timeseries_damage_counts_only_successful_lookups(damage_ratio, expected_damaged):
    pool = TimeseriesSamplePool(seed_ratio=1.0, damage_ratio=damage_ratio)
    previous_cond = torch.stack((torch.arange(7), torch.zeros(7)), dim=1)
    current_cond = torch.stack((torch.arange(20), torch.ones(20)), dim=1)
    pool.commit(torch.ones(7, 2, 4, 4), previous_cond, torch.ones(7, 1))
    pool.apply_damage = Mock(side_effect=torch.zeros_like)

    data, _, _ = pool.sample_and_replace(
        torch.full((20, 2, 4, 4), 2.0), current_cond, torch.zeros(20, 1)
    )

    # Only seven of the twenty attempted replacements have a preceding frame.
    torch.testing.assert_close(data[7:], torch.full((13, 2, 4, 4), 2.0))
    assert (data[:7] == 0).flatten(1).all(1).sum().item() == expected_damaged
    assert (data[:7] == 1).flatten(1).all(1).sum().item() == 7 - expected_damaged


@pytest.mark.parametrize(
    "seed_ratio, pool_count, frame", [(0.0, 20, 1), (1.0, 0, 1), (1.0, 20, 2)]
)
def test_timeseries_no_damage_without_reused_samples(seed_ratio, pool_count, frame):
    pool = TimeseriesSamplePool(seed_ratio=seed_ratio, damage_ratio=1.0)
    previous_cond = torch.stack((torch.arange(pool_count), torch.zeros(pool_count)), dim=1)
    pool.commit(torch.ones(pool_count, 2, 4, 4), previous_cond, torch.ones(pool_count, 1))
    pool.apply_damage = Mock(side_effect=torch.zeros_like)
    fresh = torch.full((20, 2, 4, 4), 2.0)
    current_cond = torch.stack((torch.arange(20), torch.full((20,), frame)), dim=1)

    data, cond, true = pool.sample_and_replace(fresh, current_cond, torch.zeros(20, 1))

    torch.testing.assert_close(data, fresh)
    torch.testing.assert_close(cond, current_cond)
    torch.testing.assert_close(true, torch.zeros(20, 1))
    pool.apply_damage.assert_not_called()


@pytest.mark.parametrize("class_transmute", [False, True])
def test_timeseries_forwards_pool_delay(class_transmute):
    pool = TimeseriesSamplePool(delay=5, damage_ratio=0.5, class_transmute=class_transmute)
    assert pool.delay == 5
    assert pool.mutation_ratio == 0.0

    pool.enable_seed_scheduling(total_steps=10, start_ratio=0.5, end_ratio=0.5)
    pool.enable_damage_delay(6)
    pool.apply_damage = Mock(side_effect=torch.zeros_like)
    previous_cond = torch.stack((torch.arange(20), torch.zeros(20)), dim=1)
    pool.commit(torch.ones(20, 2, 4, 4), previous_cond, torch.ones(20, 1))

    for step, expected_intact, expected_damaged in [(1, 0, 0), (4, 0, 0), (5, 10, 0), (6, 5, 5)]:
        pool.step(step)
        data, _, _ = pool.sample_and_replace(
            torch.full((20, 2, 4, 4), 2.0),
            previous_cond + torch.tensor([0, 1]),
            torch.zeros(20, 1),
        )
        assert (data == 1).flatten(1).all(1).sum().item() == expected_intact
        assert (data == 0).flatten(1).all(1).sum().item() == expected_damaged


@pytest.mark.parametrize("pool_type", [SamplePool, TimeseriesSamplePool])
def test_end_ratio_controls_final_pool_sampling_fraction(pool_type):
    pool = pool_type()
    pool.enable_seed_scheduling(total_steps=10, start_ratio=0.2, end_ratio=0.8)
    previous_cond = torch.stack((torch.arange(20), torch.zeros(20)), dim=1)
    pool.commit(torch.ones(20, 2, 4, 4), previous_cond, torch.ones(20, 1))

    for step, expected_reused in [(0, 4), (5, 10), (10, 16), (11, 16)]:
        pool.step(step)
        data, _, _ = pool.sample_and_replace(
            torch.full((20, 2, 4, 4), 2.0),
            previous_cond + torch.tensor([0, 1]),
            torch.zeros(20, 1),
        )
        assert (data == 1).flatten(1).all(1).sum().item() == expected_reused
