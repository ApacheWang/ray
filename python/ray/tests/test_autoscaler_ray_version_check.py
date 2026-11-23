"""Tests that `ray up` refuses to start Ray on a head node whose Ray version
differs from the Ray version of the `ray up` command."""
import copy
import os
import sys
from typing import List

import click
import pytest

import ray
from ray.autoscaler._private import commands
from ray.tests.autoscaler_test_utils import MockProcessRunner, MockProvider

CLUSTER_CONFIG = {
    "cluster_name": "default",
    "max_workers": 0,
    "provider": {"type": "mock"},
    "auth": {"ssh_user": "ubuntu", "ssh_private_key": os.devnull},
    "available_node_types": {
        "head": {"node_config": {}, "resources": {"CPU": 1}, "max_workers": 0},
    },
    "head_node_type": "head",
    "file_mounts": {},
    "cluster_synced_files": [],
    "initialization_commands": [],
    "setup_commands": [],
    "head_setup_commands": ["head_setup_cmd"],
    "worker_setup_commands": [],
    "head_start_ray_commands": ["start_ray_head"],
    "worker_start_ray_commands": [],
    # This test is about the version check, not token shipping.
    "auth_mode": "disabled",
}


def ray_up(version_output: List[str]) -> MockProcessRunner:
    """Run `ray up` with the head answering `ray --version` with each of
    `version_output` in turn."""
    runner = MockProcessRunner()
    runner.respond_to_call("ray --version", version_output)
    commands.get_or_create_head_node(
        copy.deepcopy(CLUSTER_CONFIG),
        printable_config_file="cluster.yaml",
        no_restart=False,
        restart_only=False,
        yes=True,
        override_cluster_name=None,
        _provider=MockProvider(),
        _runner=runner,
    )
    return runner


def command_index(runner: MockProcessRunner, pattern: str) -> int:
    for i, cmd in enumerate(runner.command_history()):
        if pattern in cmd:
            return i
    raise AssertionError(f"No command matches {pattern!r}")


def test_head_with_same_ray_version_starts_after_check():
    runner = ray_up([f"ray, version {ray.__version__}"])

    # The check runs after the setup commands, which can install Ray, and
    # before the start commands.
    setup = command_index(runner, "head_setup_cmd")
    check = command_index(runner, "ray --version")
    start = command_index(runner, "start_ray_head")
    assert setup < check < start


@pytest.mark.parametrize("node_version", ["2.0.0", ray.__version__ + "1"])
def test_head_with_other_ray_version_is_not_started(node_version):
    with pytest.raises((SystemExit, click.ClickException)):
        ray_up([f"ray, version {node_version}"])


def test_mismatch_error_names_both_versions(capsys):
    with pytest.raises((SystemExit, click.ClickException)):
        ray_up(["ray, version 2.0.0"])

    output = capsys.readouterr()
    combined = output.out + output.err
    assert "Ray version mismatch" in combined
    assert ray.__version__ in combined
    assert "2.0.0" in combined


def test_head_whose_ray_version_is_unknown_still_starts():
    runner = ray_up(["no version here"])

    runner.assert_has_call("1.2.3.4", pattern="start_ray_head")


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
