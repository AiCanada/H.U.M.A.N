# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import argparse

def str2bool(v: str) -> bool:
    """A argparse 'type' function to convert string to boolean value.

    Usage:
        parser.add_argument("--flag", type=str2bool, default=False)

        function.py --flag True/true/1/yes/y

    Args:
        v (str): String value to be converted.

    Returns:
        bool: Boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def group_values(
    parser: argparse.ArgumentParser, args: argparse.Namespace, group: str | None = None
):
    """Return {group_title: {dest: value}} for all groups on this parser."""
    out = {}
    for grp in parser._action_groups:  # private API but standard in practice
        gd = {}
        for act in grp._group_actions:
            if (
                act.dest
                and act.dest is not argparse.SUPPRESS
                and hasattr(args, act.dest)
            ):
                gd[act.dest] = getattr(args, act.dest)
        out[grp.title] = gd
    if group is None:
        return out
    else:
        return out[group]
