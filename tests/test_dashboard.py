"""Tests for the occupancy level labels shown on the dashboard."""
from db import level


def test_levels_match_the_proposal_thresholds():
    assert level(0.0)[0] == "Available"
    assert level(0.49)[0] == "Available"
    assert level(0.5)[0] == "Moderately occupied"
    assert level(0.79)[0] == "Moderately occupied"
    assert level(0.8)[0] == "Nearly full"
    assert level(0.99)[0] == "Nearly full"
    assert level(1.0)[0] == "Full"
    assert level(1.3)[0] == "Full"


def test_missing_data_is_labelled_not_coloured_as_empty():
    assert level(None)[0] == "No data"
    assert level(float("nan"))[0] == "No data"