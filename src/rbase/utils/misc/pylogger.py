# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Copyright (c) 2021 ashleve
# SPDX-License-Identifier: MIT
#
# Original file was released under MIT, with the full license text available in the folder.
#
# This modified file is released under the same license.

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import platform
import re
import sys
import threading
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Workers re-attach file handlers from this. Set by attach_run_file_logging.
RUN_LOG_DIR_ENV = "CONFROVER_RUN_LOG_DIR"
CONSOLE_LOG_NAME = "console.log"
#: CSI / OSC sequences that Rich and log_header write. Strip them from files.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)")

from lightning.pytorch.utilities import rank_zero_only

def get_pylogger(name=__name__, log_rank_zero_only: bool = True) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    if log_rank_zero_only:
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

# Common string → logging level map
LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,  # allow both "warn" and "warning"
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
    "critical": logging.CRITICAL,
    "notset": logging.NOTSET,
}

class _MaxLevelFilter(logging.Filter):
    """Allow records up to (and including) max_level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.levelno <= self.max_level

def _get_level(level: str | int) -> int:
    """
    Convert string/level to logging level int.
    Examples:
        get_level("info") -> logging.INFO
        get_level("DEBUG") -> logging.DEBUG
        get_level(20) -> 20
    """
    if isinstance(level, int):
        return level
    return LEVELS[level.lower()]

_UNICODE_LEVEL_SYMBOLS = {
    "INFO": "",
    "DEBUG": "\N{LARGE BLUE CIRCLE} ",
    "WARNING": "\N{WARNING SIGN}\N{VARIATION SELECTOR-16} ",
    "ERROR": "\N{CROSS MARK} ",
    "CRITICAL": "\N{DOUBLE EXCLAMATION MARK} ",
}

#: Plain-ASCII fallback. ⚠️ / ❌ / 🔵 / ‼️ are not in cp1252, which is the
#: default codec when stdout is redirected to a file on Windows
#: (``rbase train > train.log 2>&1``). StreamHandler.emit then raises
#: UnicodeEncodeError and logging.handleError prints an internals traceback
#: *instead of the warning*. Family filters report drops via warnings, so
#: the message the user needed was the one being destroyed.
_ASCII_LEVEL_SYMBOLS = {
    "INFO": "",
    "DEBUG": "[debug] ",
    "WARNING": "WARNING: ",
    "ERROR": "ERROR: ",
    "CRITICAL": "CRITICAL: ",
}

def configure_stdio() -> None:
    """Make stdio UTF-8 when possible; never raise UnicodeEncodeError.

    Windows ``> train.log`` is still often UTF-16 because PowerShell wraps the
    pipe. Python itself writing UTF-8 (plus ``console.log``) is what we
    control. ``errors='replace'`` is the fallback if the stream cannot switch.
    Safe to call more than once.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError, LookupError):
            try:
                reconfigure(errors="replace")
            except (OSError, ValueError, AttributeError, LookupError):
                continue

def _stream_wants_unicode_symbols(stream) -> bool:
    """Emoji prefixes only on an interactive terminal that can encode them.

    A redirected file (``isatty() is False``) always gets ASCII, even when
    Python reports UTF-8. That is the Windows ``> train.log`` case.
    """
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            if not isatty():
                return False
        except ValueError:
            return False
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        "".join(_UNICODE_LEVEL_SYMBOLS.values()).encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True

def _level_symbol_for_stream(levelname: str, stream) -> str:
    table = (
        _UNICODE_LEVEL_SYMBOLS
        if _stream_wants_unicode_symbols(stream)
        else _ASCII_LEVEL_SYMBOLS
    )
    return table.get(levelname, "")

def _level_symbol(levelname: str) -> str:
    """Prefix for a record when the handler did not bind a stream."""
    return _level_symbol_for_stream(levelname, sys.stdout)

class ConsoleFormatter(logging.Formatter):
    def __init__(self, *args, stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream = stream

    def format(self, record: logging.LogRecord) -> str:
        record.shortname = record.name.rsplit(".", 1)[-1]
        stream = self._stream
        if stream is None:
            stream = sys.stdout
        record.levelsymbol = _level_symbol_for_stream(record.levelname, stream)
        return super().format(record)

class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that never turns a UnicodeEncodeError into a traceback.

    The stdlib handler calls ``handleError``, which prints a logging-internals
    stack to stderr — and with ``> train.log 2>&1`` that *replaces* the
    warning text in the file.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                payload = (msg + self.terminator).encode(encoding, errors="replace")
                buffer = getattr(stream, "buffer", None)
                if buffer is not None:
                    buffer.write(payload)
                else:
                    stream.write(payload.decode(encoding, errors="replace"))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class _TeeTextIO:
    """Copy stdout/stderr to a UTF-8 file. ``\\r`` becomes a new line.

    PowerShell ``> file`` is UTF-16 with NULs and keeps the first ``\\r``
    frame, so a redirected train.log looks binary and the bar never moves.
    This transcript is the file to read instead.
    """

    def __init__(self, primary, file_fp) -> None:
        self._primary = primary
        self._file = file_fp

    def write(self, s) -> int:
        if not s:
            return 0
        if not isinstance(s, str):
            s = s.decode("utf-8", errors="replace")
        try:
            self._primary.write(s)
        except Exception:
            try:
                encoding = getattr(self._primary, "encoding", None) or "utf-8"
                payload = s.encode(encoding, errors="replace")
                buffer = getattr(self._primary, "buffer", None)
                if buffer is not None:
                    buffer.write(payload)
            except Exception:
                pass
        clean = _ANSI_RE.sub("", s).replace("\x00", "")
        clean = clean.replace("\r\n", "\n").replace("\r", "\n")
        if clean:
            try:
                self._file.write(clean)
                self._file.flush()
            except Exception:
                pass
        return len(s)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._primary.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(self._primary.isatty())
        except Exception:
            return False

    def fileno(self):
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", None) or "utf-8"

    @property
    def errors(self) -> str:
        return getattr(self._primary, "errors", None) or "replace"

    @property
    def buffer(self):
        return getattr(self._primary, "buffer", None)

    @property
    def closed(self) -> bool:
        return bool(getattr(self._primary, "closed", False))

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def reconfigure(self, **kwargs) -> None:
        reconfigure = getattr(self._primary, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(**kwargs)

    def __getattr__(self, name):
        return getattr(self._primary, name)

_TEE_STDOUT = None
_TEE_STDERR = None
_TEE_FILE = None
_ORIG_STDOUT = None
_ORIG_STDERR = None

def _install_console_tee(path: Path) -> None:
    """Wrap stdout/stderr so print/Rich land in ``console.log`` as UTF-8."""
    global _TEE_STDOUT, _TEE_STDERR, _TEE_FILE, _ORIG_STDOUT, _ORIG_STDERR
    if _TEE_FILE is not None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_fp = open(path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115
    except OSError:
        return
    _TEE_FILE = file_fp
    _ORIG_STDOUT = sys.stdout
    _ORIG_STDERR = sys.stderr
    _TEE_STDOUT = _TeeTextIO(sys.stdout, file_fp)
    _TEE_STDERR = _TeeTextIO(sys.stderr, file_fp)
    sys.stdout = _TEE_STDOUT
    sys.stderr = _TEE_STDERR
    _retarget_console_handlers()
    atexit.register(_flush_console_tee)

def _retarget_console_handlers() -> None:
    """Point StreamHandlers at the tee. They captured stdout at import time."""
    root = logging.getLogger("")
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if not isinstance(handler, logging.StreamHandler):
            continue
        name = getattr(handler, "_name", "")
        stream = getattr(handler, "stream", None)
        if name == "console_stderr" or stream is _ORIG_STDERR:
            handler.stream = sys.stderr
        elif name in _CONSOLE_HANDLER_NAMES or stream in (
            _ORIG_STDOUT,
            getattr(sys, "__stdout__", None),
        ):
            handler.stream = sys.stdout

def _flush_console_tee() -> None:
    fp = _TEE_FILE
    if fp is None:
        return
    try:
        fp.flush()
    except Exception:
        pass

def uninstall_console_tee() -> None:
    """Restore the original stdout/stderr. Used by tests."""
    global _TEE_STDOUT, _TEE_STDERR, _TEE_FILE, _ORIG_STDOUT, _ORIG_STDERR
    saved_out, saved_err = _ORIG_STDOUT, _ORIG_STDERR
    if saved_out is not None:
        sys.stdout = saved_out
    if saved_err is not None:
        sys.stderr = saved_err
    if _TEE_FILE is not None:
        try:
            _TEE_FILE.close()
        except Exception:
            pass
    _TEE_STDOUT = None
    _TEE_STDERR = None
    _TEE_FILE = None
    _ORIG_STDOUT = None
    _ORIG_STDERR = None
    if saved_out is not None or saved_err is not None:
        root = logging.getLogger("")
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                continue
            if not isinstance(handler, logging.StreamHandler):
                continue
            name = getattr(handler, "_name", "")
            if name == "console_stderr" and saved_err is not None:
                handler.stream = saved_err
            elif name in _CONSOLE_HANDLER_NAMES and saved_out is not None:
                handler.stream = saved_out

def dataloader_worker_init(worker_id: int) -> None:
    """Re-attach file logs in a spawned DataLoader worker (Windows spawn)."""
    raw = os.environ.get(RUN_LOG_DIR_ENV)
    if not raw:
        return
    attach_run_file_logging(
        Path(raw),
        command=f"dataloader-worker-{int(worker_id)}",
        append=True,
        quiet=True,
    )

class SafeFileHandler(logging.FileHandler):
    """FileHandler that always writes UTF-8 and never swallows a record."""

    def __init__(self, filename, mode: str = "a") -> None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode=mode, encoding="utf-8", delay=False)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            try:
                payload = (
                    f"{record.asctime if hasattr(record, 'asctime') else ''} "
                    f"{record.levelname} {record.getMessage()}\n"
                )
                self.stream.write(payload)
                if record.exc_info:
                    self.stream.write(
                        "".join(traceback.format_exception(*record.exc_info))
                    )
                self.flush()
            except Exception:
                self.handleError(record)

def _file_formatter() -> logging.Formatter:
    return logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)

def _drop_run_file_handlers(logger: logging.Logger) -> None:
    stale = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_name", "") in _FILE_HANDLER_NAMES
    ]
    for handler in stale:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

def attach_run_file_logging(
    log_dir,
    *,
    command: str | None = None,
    append: bool = True,
    quiet: bool = False,
) -> Path:
    """Write every record to disk under ``log_dir``.

    * ``debug.log`` — DEBUG and above (full train/generate trace).
    * ``issues.log`` — WARNING and above (filters, CUDA, dead units, crashes).
    * ``console.log`` — UTF-8 transcript of stdout/stderr (print, Rich, Lightning).
    * ``faults.log`` — native fatal signals via faulthandler.
    * ``environment.txt`` — argv, Python, torch/CUDA, git (written once).

    ``quiet=True`` is for DataLoader workers: handlers only, no banner, no
    tee (the parent already owns stdout). Safe to call more than once.
    """
    configure_stdio()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ[RUN_LOG_DIR_ENV] = str(log_dir.resolve())
    mode = "a" if append else "w"

    root = logging.getLogger("")
    root.setLevel(logging.DEBUG)
    _drop_run_file_handlers(root)

    debug_path = log_dir / "debug.log"
    issues_path = log_dir / "issues.log"
    h_debug = SafeFileHandler(debug_path, mode=mode)
    h_debug.setLevel(logging.DEBUG)
    h_debug.setFormatter(_file_formatter())
    h_debug._name = "run_debug_file"
    root.addHandler(h_debug)

    h_issues = SafeFileHandler(issues_path, mode=mode)
    h_issues.setLevel(logging.WARNING)
    h_issues.setFormatter(_file_formatter())
    h_issues._name = "run_issues_file"
    root.addHandler(h_issues)

    install_debug_hooks(fault_path=log_dir / "faults.log")
    if quiet:
        return log_dir

    _install_console_tee(log_dir / CONSOLE_LOG_NAME)
    _write_environment(log_dir / "environment.txt", command=command)

    banner = (
        f"==== file logging {datetime.now(timezone.utc).isoformat()} "
        f"command={command or ' '.join(sys.argv)!r} dir={log_dir} ===="
    )
    root.info(banner)
    root.info(f"debug log: {debug_path.resolve()}")
    root.info(f"issues log (WARNING+): {issues_path.resolve()}")
    root.info(f"console log (stdout/stderr UTF-8): {(log_dir / CONSOLE_LOG_NAME).resolve()}")
    return log_dir

#: Matched with re.escape as a prefix. These are not RBase bugs.
_IGNORED_WARNING_PREFIXES: tuple[str, ...] = (
    "`isinstance(treespec, LeafSpec)` is deprecated",
)

def _ignore_known_upstream_warnings() -> None:
    """Re-apply after any hook that resets the warnings filter."""
    for prefix in _IGNORED_WARNING_PREFIXES:
        warnings.filterwarnings("ignore", message=re.escape(prefix))

def install_debug_hooks(fault_path=None) -> None:
    """Route warnings and uncaught exceptions through logging (once)."""
    global _HOOKS_INSTALLED, _FAULT_FILE
    logging.captureWarnings(True)
    # Do not simplefilter("default") for all DeprecationWarnings: that undoes
    # the LeafSpec ignore and reprints Lightning's internal torch.pytree
    # warning on every Trainer() construct.

    if fault_path is not None:
        try:
            if _FAULT_FILE is not None:
                try:
                    _FAULT_FILE.close()
                except Exception:
                    pass
            fault_file = open(fault_path, "a", encoding="utf-8")  # noqa: SIM115
            faulthandler.enable(file=fault_file, all_threads=True)
            _FAULT_FILE = fault_file
        except OSError:
            try:
                faulthandler.enable(all_threads=True)
            except Exception:
                pass

    _ignore_known_upstream_warnings()
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        logging.getLogger("rbase").error(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
        _orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):
        _orig_thread = threading.excepthook

        def _thread_hook(args):
            logging.getLogger("rbase").error(
                f"Uncaught thread exception in {args.thread}",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _orig_thread(args)

        threading.excepthook = _thread_hook

def _write_environment(path: Path, *, command: str | None) -> None:
    lines = [
        f"utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command or ' '.join(sys.argv)}",
        f"cwd={os.getcwd()}",
        f"pid={os.getpid()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
    ]
    for key in sorted(k for k in os.environ if k.startswith("CONFROVER_")):
        lines.append(f"env.{key}={os.environ[key]}")
    try:
        import torch

        lines.append(f"torch={torch.__version__}")
        lines.append(f"cuda_available={torch.cuda.is_available()}")
        lines.append(f"cuda_device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            lines.append(f"cuda={torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                lines.append(
                    f"gpu[{i}]={props.name} "
                    f"mem_bytes={props.total_memory}"
                )
    except Exception as exc:
        lines.append(f"torch_probe_failed={type(exc).__name__}: {exc}")
    try:
        import lightning

        lines.append(f"lightning={lightning.__version__}")
    except Exception:
        pass
    try:
        import transformers

        lines.append(f"transformers={transformers.__version__}")
    except Exception:
        pass
    try:
        import subprocess

        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if git.returncode == 0:
            lines.append(f"git={git.stdout.strip()}")
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if dirty.returncode == 0:
            lines.append(f"git_dirty={bool(dirty.stdout.strip())}")
    except Exception:
        pass
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

_CONSOLE_HANDLER_NAMES = {
    "console_stdout",
    "console_stderr",
    "console_stdout_all",
}
_FILE_HANDLER_NAMES = {
    "run_debug_file",
    "run_issues_file",
}

#: Full detail for a file a human (or a later session) will grep.
_FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s "
    "[pid=%(process)d tid=%(thread)d %(name)s %(filename)s:%(lineno)d %(funcName)s] "
    "%(message)s"
)
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"

_HOOKS_INSTALLED = False
_FAULT_FILE = None

def _has_handler(logger: logging.Logger, name: str) -> bool:
    return any(getattr(h, "_name", None) == name for h in logger.handlers)

def _drop_legacy_console_handlers(logger: logging.Logger) -> None:
    """Replace stdlib StreamHandlers left over from an older setup_logging."""
    stale = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_name", "") in _CONSOLE_HANDLER_NAMES
        and not isinstance(handler, SafeStreamHandler)
    ]
    for handler in stale:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

def set_console_level(level: int, *, logger: Optional[logging.Logger] = None) -> None:
    """Change console handler levels (stdout and/or stderr) together."""
    lg = logger or logging.getLogger("")
    for h in lg.handlers:
        if getattr(h, "_name", "") in {
            "console_stdout",
            "console_stderr",
            "console_stdout_all",
        }:
            # stdout handler keeps DEBUG level to allow MaxLevelFilter; stderr level is WARNING anyway.
            if getattr(h, "_name", "") == "console_stdout_all":
                h.setLevel(level)

def setup_logging(
    *,
    console_level: int | str = logging.INFO,
    split_stderr: bool = False,
    console_format: str = "%(levelsymbol)s%(message)s",
    # datefmt: Optional[str] = None,
    root_name: str = "",  # "" = root logger
    propagate: bool = False,  # set False for library packages
) -> logging.Logger:
    """
    Configure clean console logging once. Safe to call multiple times.

    Parameters
    ----------
    console_level : int
        Logging level for console (stdout/stderr) handlers.
    split_stderr : bool
        If True, INFO and below go to stdout; WARNING and above go to stderr.
        If False, everything goes to stdout.
    console_format : str
        Minimal, readable console formatter. Default: "[I] message".
    datefmt : Optional[str]
        Date format for console. None keeps console timestamp-free by default.
    root_name : str
        Logger name to attach handlers to ("" is the root logger).
    propagate : bool
        If False, prevent double logs when users also configure logging.

    Returns
    -------
    logging.Logger
        The configured base logger.
    """
    configure_stdio()
    base = logging.getLogger(root_name)
    base.setLevel(logging.DEBUG)  # capture everything; handlers filter what to show
    base.propagate = propagate
    console_level = _get_level(console_level)
    _drop_legacy_console_handlers(base)

    # stdout handler (INFO and below)
    if split_stderr:
        if not _has_handler(base, "console_stdout"):
            h_out = SafeStreamHandler(stream=sys.stdout)
            h_out.setLevel(logging.DEBUG)  # accept all; filter limits
            h_out.addFilter(_MaxLevelFilter(logging.INFO))
            h_out.setFormatter(
                ConsoleFormatter(fmt=console_format, stream=sys.stdout)
            )
            h_out._name = "console_stdout"
            base.addHandler(h_out)

        # stderr handler (WARNING and above)
        if not _has_handler(base, "console_stderr"):
            h_err = SafeStreamHandler(stream=sys.stderr)
            h_err.setLevel(logging.WARNING)
            h_err.setFormatter(
                ConsoleFormatter(fmt=console_format, stream=sys.stderr)
            )
            h_err._name = "console_stderr"
            base.addHandler(h_err)
    else:
        if not _has_handler(base, "console_stdout_all"):
            h_all = SafeStreamHandler(stream=sys.stdout)
            h_all.setLevel(console_level)
            h_all.setFormatter(
                ConsoleFormatter(fmt=console_format, stream=sys.stdout)
            )
            h_all._name = "console_stdout_all"
            base.addHandler(h_all)

    # Adjust levels after handlers exist
    set_console_level(console_level, logger=base)

    return base

RESET = "\x1b[0m"
BOLD = "\x1b[1m"

def _supports_color(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()

def _rule(text: str, width: int = 64, char: str = "=") -> str:
    text = f" {text} "
    fill = max(0, width - len(text)) // 2
    return f"{char * fill} {text}{char * fill}"

def log_header(logger: logging.Logger, title: str, char="="):
    stream = next(
        (h.stream for h in logger.handlers if hasattr(h, "stream")), sys.stderr
    )
    line = _rule(title, char=char)
    if _supports_color(stream):
        line = f"\n{BOLD}{line}{RESET}"
    return line

setup_logging(console_level="info")
