"""No file that ships carries a machine's real name.

There used to be two tables --- one in `scripts/hosts.py`, one in
`rust/qomm-measure/src/hosts.rs` --- and a test here that held them to each
other. It did that job and missed the point entirely: both files ship, so both
published the names the labels exist to withhold, and both had done so since the
first commit.

There is now one table, in `scripts/host_map.txt`, which is not exported. So the
test worth having is not that two copies agree. It is that no copy exists in
anything that leaves this tree, and that is checked against the local table
rather than against a list somebody keeps up to date by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.hosts import LABELS, load                              # noqa: E402

RUST = ROOT / "rust" / "qomm-measure" / "src" / "hosts.rs"
PYTHON = ROOT / "scripts" / "hosts.py"

# Where the names must not appear. `artifacts/` is included because that is
# where they used to be, and `scripts/host_map.txt` is the one file exempt ---
# it is the table, and it does not ship.
SHIPS = ("scripts", "rust", "qomm_sim", "qomm_dsl", "qomm_audit",
         "qomm_transport", "qomm_demo", "zk", "defmi", "mp_spdz", "tests",
         "artifacts", "notebooks")
EXEMPT = {ROOT / "scripts" / "host_map.txt"}
SKIP_DIRS = {"target", "__pycache__", ".pytest_cache", "lib", "out", "cache",
             "broadcast", "superseded"}

# A published copy has no table, which is the whole point, so the checks that
# need one skip there rather than fail. The structural checks below do not need
# it and run everywhere --- those are the ones that would catch a second copy
# growing back.
MAP = ROOT / "scripts" / "host_map.txt"
needs_the_table = pytest.mark.skipif(
    not MAP.exists(),
    reason="no local table, so this is a published copy --- which is the point")


def files_that_ship():
    for top in SHIPS:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path in EXEMPT:
                continue
            if SKIP_DIRS & set(path.relative_to(ROOT).parts):
                continue
            yield path


@needs_the_table
def test_the_table_is_somewhere_that_does_not_ship():
    assert LABELS, "the local table is there but empty; nothing would be labelled"
    assert MAP.exists()


@needs_the_table
@pytest.mark.parametrize("name", sorted(LABELS) or ["(no table here)"])
def test_no_real_machine_name_appears_in_a_file_that_ships(name):
    """The check that would have caught it, run against every name we hold."""
    guilty = []
    for path in files_that_ship():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if name in text:
            guilty.append(str(path.relative_to(ROOT)))
    assert not guilty, f"{name} appears in {', '.join(guilty[:5])}"


def test_neither_reader_holds_a_table_of_its_own():
    """A second copy is how this went wrong, so its absence is asserted."""
    assert "pub const LABELS" not in RUST.read_text(encoding="utf-8")
    python = PYTHON.read_text(encoding="utf-8")
    assert "LABELS = load()" in python
    assert "LABELS = {" not in python


def test_both_readers_look_in_the_same_place():
    rust = RUST.read_text(encoding="utf-8")
    python = PYTHON.read_text(encoding="utf-8")
    for token in ("QOMM_HOST_MAP", "host_map.txt"):
        assert token in rust, token
        assert token in python, token


def test_the_lookup_takes_the_exact_name_then_the_short_one(tmp_path):
    """Stated against machines that do not exist, for the obvious reason."""
    table = tmp_path / "map.txt"
    table.write_text("# a comment\n\ngrinder      site-one  # and another\n"
                     "kettle.local site-two\n", encoding="utf-8")
    found = load(table)
    assert found == {"grinder": "site-one", "kettle.local": "site-two"}

    def label(node: str) -> str:
        return found.get(node, found.get(node.split(".")[0], node))

    assert label("grinder") == "site-one"
    assert label("grinder.internal") == "site-one"
    assert label("kettle.local") == "site-two"
    # listed with `.local`, so its bare form is not in the table
    assert label("kettle") == "kettle"
    assert label("somebody-elses-laptop") == "somebody-elses-laptop"


def test_a_missing_table_labels_nothing_rather_than_failing(tmp_path):
    """What a reader of the published repository gets, and it has to work."""
    assert load(tmp_path / "not-here.txt") == {}


def test_the_rust_reader_parses_the_format_the_python_writes():
    """Both drop a `#` comment, split on whitespace, and skip blank lines."""
    rust = RUST.read_text(encoding="utf-8")
    assert re.search(r"split\('#'\)", rust)
    assert "split_whitespace" in rust
