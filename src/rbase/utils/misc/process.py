# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
from concurrent.futures import TimeoutError
from functools import partial
from subprocess import PIPE, Popen
from typing import Callable, Optional

from tqdm import tqdm

# from pebble import ProcessPool, ProcessExpired
from .pylogger import get_pylogger

logger = get_pylogger(__name__)

def check_exec(exec_path=None, env=None, default=None, error_suffix=None) -> str:
    """Check if an executable exists

    Returns:
        the path to the executable if exists
    """
    if exec_path is not None:
        exec_cmd = str(exec_path)
    elif env is not None:
        exec_cmd = os.environ.get(env, default)
        if exec_cmd is None:
            raise RuntimeError(
                f"Environment variable {env} not set, and no default exec provided.\n{error_suffix}"
            )
    else:
        assert default is not None, f"No command info found, please check the code."
        exec_cmd = default

    if shutil.which(exec_cmd) is None:
        raise RuntimeError(
            f"Command `{exec_cmd}` not found. Check if one of the following is provided:\n"
            f"   exec_path: {exec_path}\n   env {env}: {os.environ.get(str(env), None)}\n   default: {default}\n{error_suffix}"
        )
    return str(exec_cmd)

def subprocess_run(cmd, env=None, quiet=False, cwd=None, prefix=">>> "):
    """Run shell subprocess"""

    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, env=env, cwd=cwd)
    out = ""

    for line in iter(proc.stdout.readline, b""):  # type:ignore
        out_line = line.decode("utf-8").rstrip()
        out += out_line
        if not quiet:
            print("{}{}".format(prefix, out_line), flush=True)
    stdout, stderr = proc.communicate()
    err = stderr.decode("utf-8").strip("\n")
    return out, err

def mp_imap_unordered(
    iter, func, n_proc: int = 1, chunksize: int = 1, mute_tqdm: bool = False, **kwargs
):
    """Helper function for multiprocessing run. The each item in the iterable should contain only one argument"""

    if len(kwargs) > 0:
        func = partial(func, **kwargs)
    n_proc = min(n_proc, len(iter))
    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            try:
                results = list(
                    tqdm(
                        p.imap_unordered(func, iter, chunksize=chunksize),
                        total=len(iter),
                        disable=mute_tqdm,
                    )
                )
                p.close()
                p.terminate()
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, terminating processes..", flush=True)
                p.terminate()
                p.join()
                results = []
    else:
        results = [func(x) for x in tqdm(iter, disable=mute_tqdm)]

    return results

def mp_imap(
    iter, func, n_proc: int = 1, chunksize: int = 1, mute_tqdm: bool = False, **kwargs
):
    """Helper function for multiprocessing run. The each item in the iterable should contain only one argument"""

    if len(kwargs) > 0:
        func = partial(func, **kwargs)

    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            try:
                results = list(
                    tqdm(
                        p.imap(func, iter, chunksize=chunksize),
                        total=len(iter),
                        disable=mute_tqdm,
                    )
                )
                p.close()
                p.terminate()
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, terminating processes..", flush=True)
                p.terminate()
                p.join()
                results = []
    else:
        results = [func(x) for x in tqdm(iter, disable=mute_tqdm)]

    return results
