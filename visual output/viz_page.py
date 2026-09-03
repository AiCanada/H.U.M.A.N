# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Splice a payload into the viewer template and check the result.

The output is one file with no external dependency but three.js from a CDN, so
it can be mailed, dropped on a share, or opened from a USB stick years from now
and still work. That is the whole reason the coordinates are embedded rather
than fetched.

Three checks run before anything is written, because a page like this is made to
be handed to other people:

  * the payload cannot close its own ``<script>`` element -- the one sequence
    that would turn embedded data into executable markup;
  * every element that opens closes, so a page that silently renders blank is
    caught here rather than in front of an audience;
  * anything named in ``forbid`` is absent. Coordinates are innocuous; the
    headers around them are not always. A CASP submission carries the group's
    registration code in every model's AUTHOR record, and that is a credential.
    The check is a parameter rather than a fixed list because what must not
    leave the machine is a property of the data, not of this code.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")

_TAGS = ("div", "script", "style", "header", "svg", "span", "p", "input", "button")
_VOID = {"input", "link", "br", "img", "meta"}


def render(payload: dict, *, template: Path | None = None,
           forbid=()) -> str:
    """The finished page as a string."""
    tpl = (template or TEMPLATE).read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":"))

    if "/*__PAYLOAD__*/" not in tpl or "__TITLE__" not in tpl:
        raise SystemExit(f"{template or TEMPLATE}: template has lost its placeholders")
    # The payload sits inside <script type="application/json">; the only sequence
    # that could close it early is a literal </script>, which JSON never
    # produces from these fields -- but the check is cheap and the failure mode
    # is arbitrary markup injection, so it is asserted rather than assumed.
    if "</script" in blob.lower():
        raise SystemExit("payload contains '</script'; refusing to embed it")

    page = tpl.replace("__TITLE__", html.escape(str(payload["meta"]["title"])))
    page = page.replace("/*__PAYLOAD__*/", blob)

    for bad in forbid:
        if bad and bad in page:
            raise SystemExit(f"refusing to write the page: it contains {bad!r}")

    for tag in _TAGS:
        if tag in _VOID:
            continue
        opened = len(re.findall(rf"<{tag}[\s>]", page))
        closed = len(re.findall(rf"</{tag}>", page))
        if opened != closed:
            raise SystemExit(f"unbalanced <{tag}>: {opened} open, {closed} close")
    return page


def write(payload: dict, out: Path, *, template: Path | None = None,
          forbid=()) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload, template=template, forbid=forbid), encoding="utf-8")
    return out
