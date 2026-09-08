"""Red gate: the charter must pin *which* containment receipt it means.

Approval reads the receipt file and binds its bytes into the anchor, which stops
the receipt changing after approval.  It does not stop the receipt being swapped
*before* approval: whatever file sits at `receipt_path` at that moment is what
gets bound, and the charter has said nothing about which file it expected.

`receipt_sha256` closes that window -- but only if the runtime reads it.  Until
now it did not: the field existed in `evidence/bundle1/charter.json` and appeared
nowhere in `loop/*.py`, so it was decoration that looked like a commitment.
"""
import json
import os

import pytest

import preflight


HERE = os.path.dirname(os.path.abspath(__file__))
DECLARATION = os.path.realpath(
    os.path.join(HERE, os.pardir, "evidence", "bundle1", "receipt-declaration.json"))


def _declaration():
    return json.load(open(DECLARATION))


def _charter_for(receipt_path, declared_sha):
    negative_test = {"receipt_path": receipt_path}
    if declared_sha is not None:
        negative_test["receipt_sha256"] = declared_sha
    return {"containment": {"negative_test": negative_test}}


def test_declared_receipt_digest_is_required(tmp_path):
    """A receipt named without a digest is a receipt nobody committed to."""
    declaration = _declaration()
    charter = _charter_for(declaration["receipt_path"], None)

    value, problem = preflight._read_containment_receipt(
        charter, declaration["targets"])

    assert value is None
    assert problem and "receipt_sha256" in problem, problem


def test_declared_receipt_digest_must_match_the_bytes_on_disk(tmp_path):
    """The swap this field exists to catch: right shape, wrong receipt."""
    declaration = _declaration()
    real = open(declaration["receipt_path"], "rb").read()
    swapped = tmp_path / "negative_test_receipts.json"
    value = json.loads(real.decode())
    value["at"] = "1999-01-01T00:00:00+09:00"     # a different, still valid receipt
    swapped.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")

    charter = _charter_for(os.path.realpath(str(swapped)),
                           declaration["receipt_sha256"])

    bound, problem = preflight._read_containment_receipt(
        charter, declaration["targets"])

    assert bound is None
    assert problem and "does not match" in problem, problem


def test_the_declared_receipt_itself_is_accepted():
    """The control: the real declaration must still bind, or the check is a wall.

    This is the half that would be missing if the test only proved refusals --
    a `_read_containment_receipt` that rejected everything would pass the two
    tests above.
    """
    declaration = _declaration()
    charter = _charter_for(declaration["receipt_path"],
                           declaration["receipt_sha256"])

    bound, problem = preflight._read_containment_receipt(
        charter, declaration["targets"])

    assert problem is None, problem
    assert bound["sha256"] == declaration["receipt_sha256"]


def test_declaration_pins_the_receipt_actually_on_disk():
    """The declaration must describe this checkout, not a receipt long gone."""
    declaration = _declaration()
    raw = open(declaration["receipt_path"], "rb").read()

    import hashlib
    assert hashlib.sha256(raw).hexdigest() == declaration["receipt_sha256"]


def test_parser_refuses_provenance_whose_cgroup_does_not_name_its_unit(tmp_path):
    """Three non-empty strings are not a binding.

    The cgroup path is minted by the kernel and ends in the unit that owns it,
    so `cgroup` and `unit` disagreeing means the receipt is describing two
    different things. Checking that the checked-in file happens to be
    consistent -- which the test below does -- proves nothing about whether an
    inconsistent one would be refused, and only the second is a property.
    """
    declaration = _declaration()
    value = json.loads(open(declaration["receipt_path"], "rb").read().decode())
    value["provenance"]["contained"]["unit"] = "some-other.service"
    swapped = tmp_path / "negative_test_receipts.json"
    swapped.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    path = os.path.realpath(str(swapped))

    # Re-pin, or this is refused as a swapped receipt and never reaches the
    # provenance check it is named for.
    charter = _charter_for(path, preflight.file_digest(path))

    bound, problem = preflight._read_containment_receipt(
        charter, declaration["targets"])

    assert bound is None
    assert problem and "cgroup" in problem and "unit" in problem, problem


def test_contained_half_carries_execution_provenance():
    """The contained outcomes must come from something that ran contained.

    Without this the receipt is four `blocked` strings, which any editor can
    type. The cgroup path and invocation id are minted by systemd and the
    kernel, so they are the part of the receipt a mistaken hand cannot supply.
    """
    receipt = json.load(open(_declaration()["receipt_path"]))
    contained = receipt["provenance"]["contained"]

    assert contained["invocation_id"], "contained half has no systemd invocation id"
    assert contained["cgroup"].endswith("/" + contained["unit"]), contained
    assert all(outcome["contained"] == "blocked"
               for outcome in receipt["probes"].values())
    assert all(outcome["control"] == "succeeded"
               for outcome in receipt["probes"].values())
