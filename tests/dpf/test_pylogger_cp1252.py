# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Windows ``> train.log`` uses cp1252. Level emoji must not wipe warnings."""

from __future__ import annotations

import io
import logging

from rbase.utils.misc.pylogger import (
    ConsoleFormatter,
    SafeStreamHandler,
    _ASCII_LEVEL_SYMBOLS,
    _UNICODE_LEVEL_SYMBOLS,
    _stream_wants_unicode_symbols,
)

# The message family filters actually emit — this is what used to vanish.
_FILTER_WARNING = (
    "Dropping 13 of 100 families longer than --max_seqlen 384: the fused "
    "token axis is L+L^2, so these are what exhausts GPU memory: 1x54_A (L=434)"
)

def _cp1252_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="\n")
    return raw, stream

def _attach(stream: io.TextIOWrapper) -> logging.Logger:
    logger = logging.getLogger("rbase.test.cp1252")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = SafeStreamHandler(stream)
    handler.setFormatter(
        ConsoleFormatter(fmt="%(levelsymbol)s%(message)s", stream=stream)
    )
    logger.addHandler(handler)
    return logger

def test_family_filter_warning_survives_cp1252_redirect():
    raw, stream = _cp1252_stream()
    logger = _attach(stream)

    logger.warning(_FILTER_WARNING)
    stream.flush()
    text = raw.getvalue().decode("cp1252")

    assert _FILTER_WARNING in text
    assert text.startswith("WARNING:")
    assert "Traceback" not in text
    assert "UnicodeEncodeError" not in text
    assert "\u26a0" not in text
    assert "\u274c" not in text

def test_attach_run_file_logging_writes_debug_and_issues(tmp_path):
    import warnings

    from rbase.utils.misc.pylogger import (
        attach_run_file_logging,
        uninstall_console_tee,
    )

    log_dir = tmp_path / "logs"
    attach_run_file_logging(log_dir, command="train")
    root = logging.getLogger("")
    root.debug("probe debug detail")
    root.info("probe info heartbeat")
    root.warning(_FILTER_WARNING)
    try:
        raise RuntimeError("synthetic crash for issues.log")
    except RuntimeError:
        logging.getLogger("rbase.test").exception("trainer.fit failed")
    warnings.warn("future probe", FutureWarning)

    debug = (log_dir / "debug.log").read_text(encoding="utf-8")
    issues = (log_dir / "issues.log").read_text(encoding="utf-8")
    env = (log_dir / "environment.txt").read_text(encoding="utf-8")

    assert "probe debug detail" in debug
    assert "probe info heartbeat" in debug
    assert _FILTER_WARNING in debug
    assert "synthetic crash" in debug
    assert "Traceback" in debug
    assert _FILTER_WARNING in issues
    assert "synthetic crash" in issues
    assert "Traceback" in issues
    assert "probe debug detail" not in issues
    assert "command=train" in env
    assert "python=" in env
    from rbase.utils.misc.pylogger import _drop_run_file_handlers

    _drop_run_file_handlers(logging.getLogger(""))
    uninstall_console_tee()

def test_leafspec_deprecation_is_not_logged_after_hooks(tmp_path):
    """Lightning constructs LeafSpec(); torch 2.13 warns. Not our bug."""
    import warnings

    from rbase.utils.misc.pylogger import (
        _drop_run_file_handlers,
        attach_run_file_logging,
        uninstall_console_tee,
    )

    attach_run_file_logging(tmp_path / "logs", command="train")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from rbase.utils.misc.pylogger import _ignore_known_upstream_warnings

        _ignore_known_upstream_warnings()
        warnings.warn(
            "`isinstance(treespec, LeafSpec)` is deprecated, use "
            "`isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.",
            DeprecationWarning,
            stacklevel=1,
        )
    assert not any("LeafSpec" in str(w.message) for w in caught)
    _drop_run_file_handlers(logging.getLogger(""))
    uninstall_console_tee()

def test_emoji_in_message_does_not_replace_warning_with_traceback():
    raw, stream = _cp1252_stream()
    logger = _attach(stream)

    logger.warning("\u26a0\ufe0f leftover emoji then " + _FILTER_WARNING)
    stream.flush()
    text = raw.getvalue().decode("cp1252")

    assert _FILTER_WARNING in text
    assert "Traceback" not in text
    assert "UnicodeEncodeError" not in text

def test_redirected_stream_uses_ascii_prefix_even_if_utf8():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="\n")
    assert stream.isatty() is False
    assert _stream_wants_unicode_symbols(stream) is False
    logger = _attach(stream)
    logger.warning(_FILTER_WARNING)
    stream.flush()
    text = raw.getvalue().decode("utf-8")
    assert text.startswith(_ASCII_LEVEL_SYMBOLS["WARNING"])
    assert _UNICODE_LEVEL_SYMBOLS["WARNING"] not in text

def test_utf8_tty_keeps_emoji_prefix():
    class _Tty(io.TextIOWrapper):
        def isatty(self) -> bool:
            return True

    stream = _Tty(io.BytesIO(), encoding="utf-8", errors="strict")
    assert _stream_wants_unicode_symbols(stream) is True
    assert _stream_wants_unicode_symbols(
        io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    ) is False

def test_stdlib_handler_on_cp1252_is_what_we_are_fixing():
    """Document the failure mode: stdlib emit + emoji prefix + cp1252."""
    raw, stream = _cp1252_stream()
    logger = logging.getLogger("rbase.test.cp1252.stdlib")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)

    class _EmojiFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelsymbol = _UNICODE_LEVEL_SYMBOLS.get(record.levelname, "")
            return super().format(record)

    handler.setFormatter(_EmojiFormatter("%(levelsymbol)s%(message)s"))
    logger.addHandler(handler)

    errors: list[str] = []

    def _handle_error(record: logging.LogRecord) -> None:
        errors.append(record.getMessage())

    handler.handleError = _handle_error  # type: ignore[method-assign]
    logger.warning(_FILTER_WARNING)
    stream.flush()
    text = raw.getvalue().decode("cp1252")

    assert errors == [_FILTER_WARNING]
    assert _FILTER_WARNING not in text

def test_console_tee_is_utf8_and_turns_cr_into_newlines(tmp_path):
    """PowerShell > file is UTF-16 + \\r; console.log must be a real transcript."""
    import sys

    from rbase.utils.misc.pylogger import (
        CONSOLE_LOG_NAME,
        _drop_run_file_handlers,
        attach_run_file_logging,
        uninstall_console_tee,
    )

    log_dir = tmp_path / "logs"
    try:
        attach_run_file_logging(log_dir, command="train")
        logging.getLogger("").info("heartbeat via logging")
        print("hello\rEpoch 1/90 [>] 10/1216")
        sys.stdout.write("\x1b[1mbold\x1b[0m print\n")
        sys.stdout.flush()
        text = (log_dir / CONSOLE_LOG_NAME).read_text(encoding="utf-8")
    finally:
        _drop_run_file_handlers(logging.getLogger(""))
        uninstall_console_tee()

    assert "\x00" not in text
    assert "\r" not in text
    assert "hello" in text
    assert "Epoch 1/90" in text
    assert "bold print" in text
    assert "heartbeat via logging" in text
    assert "\x1b" not in text
    debug = (log_dir / "debug.log").read_text(encoding="utf-8")
    assert "console log" in debug

def test_quiet_worker_attach_does_not_rewrite_environment(tmp_path):
    from rbase.utils.misc.pylogger import (
        _drop_run_file_handlers,
        attach_run_file_logging,
        dataloader_worker_init,
        uninstall_console_tee,
    )

    log_dir = tmp_path / "logs"
    try:
        attach_run_file_logging(log_dir, command="train")
        env_first = (log_dir / "environment.txt").read_text(encoding="utf-8")
        dataloader_worker_init(3)
        env_after = (log_dir / "environment.txt").read_text(encoding="utf-8")
        debug = (log_dir / "debug.log").read_text(encoding="utf-8")
    finally:
        _drop_run_file_handlers(logging.getLogger(""))
        uninstall_console_tee()

    assert env_first == env_after
    assert debug.count("==== file logging") == 1
