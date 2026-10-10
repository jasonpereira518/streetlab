"""The TS and Python key lists are hand-maintained twins; read both so they cannot drift."""

import re
from pathlib import Path
from typing import get_args

from schema import LayerKey
from sim.loop import DEFAULT_PARAMS

SRC = Path(__file__).resolve().parents[2] / "streetlab" / "src"


def test_layer_keys_match_the_zod_enum():
    ts = (SRC / "schema.ts").read_text()
    body = re.search(r"LayerKeySchema = z\.enum\(\[(.*?)\]\)", ts, re.S).group(1)
    assert set(re.findall(r"'(\w+)'", body)) == set(get_args(LayerKey))


def test_every_server_param_slider_is_honoured_by_the_backend():
    """A non-clientOnly control in PARAM_DEFS that the backend ignores is a dead control."""
    ts = (SRC / "store" / "simStore.ts").read_text()
    defs = ts.split("PARAM_DEFS")[1].split("\n];")[0]
    server_keys = {
        m.group(1)
        for block in defs.split("\n  {\n")[1:]
        if "clientOnly" not in block and (m := re.search(r"key: '(\w+)'", block))
    }
    assert server_keys == set(DEFAULT_PARAMS)
