"""Parameter policy: every config value tagged and sourced; SWEEPs are lists; STD values cite a source."""
from shieldmap import config, grid


def test_config_satisfies_policy():
    assert config.validate() == []


def test_every_leaf_tagged():
    n = 0
    for key, leaf in config.iter_leaves(config.raw()):
        assert leaf["tag"] in config.TAGS, key
        n += 1
    assert n > 20


def test_no_cross_criterion_weights():
    keys = [k for k, _ in config.iter_leaves(config.raw())]
    assert not any("weight" in k.lower() for k in keys)


def test_grid_contains_lines_and_anchors():
    g = grid.energy_grid_keV()
    for e in (10.0, 60.0, 100.0, 200.0, 600.0, 800.0, 1000.0, 15000.0, *grid.key_lines_keV()):
        assert (abs(g - e) < 1e-9).any(), e
    assert len(grid.key_lines_keV()) == 3
    assert (g[1:] > g[:-1]).all()
