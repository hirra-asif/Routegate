"""Preserve historical veto decisions while retiring the guarded pilot."""
import csv
import hashlib
import json

import pytest

from routegate import historicalveto as V
from routegate.validation import ValidationError


@pytest.fixture
def veto_inputs(tmp_path, monkeypatch):
    records = [
        ('permit', 0.1, 0.1),
        ('on_residual', 0.2, 0.1),
        ('on_routing', 0.1, 0.3),
    ]
    pins = {}
    for source in V.VETO_SOURCES:
        pred = source.prediction_path(tmp_path)
        thr = source.threshold_file(tmp_path)
        pred.parent.mkdir(parents=True, exist_ok=True)
        with pred.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_id', source.probability_column])
            index = 1 if source.family == 'residual' else 2
            writer.writerows((row[0], row[index]) for row in records)
        document = ({'threshold_H': 0.2} if source.family == 'residual' else
                    {'families': {'routing': {'threshold_H': {'threshold': 0.3}}}})
        thr.write_text(json.dumps(document))
        pins[source.predictions_digest_key] = hashlib.sha256(pred.read_bytes()).hexdigest()
        pins[source.thresholds_digest_key] = hashlib.sha256(thr.read_bytes()).hexdigest()
    monkeypatch.setattr(V, 'EXPECTED_INPUT_DIGESTS', pins)
    return tmp_path


def test_threshold_boundary_remains_vetoed(veto_inputs):
    readings, _ = V.load_hard_vetoes(root=veto_inputs)
    assert readings['permit'].permits
    assert not readings['on_residual'].permits
    assert not readings['on_routing'].permits


def test_changed_sensor_record_is_rejected_before_use(veto_inputs):
    path = V.RESIDUAL_VETO.prediction_path(veto_inputs)
    path.write_text(path.read_text().replace('0.2', '0.1'))
    with pytest.raises(ValidationError, match='hash mismatch'):
        V.load_hard_vetoes(root=veto_inputs)


def test_missing_sensor_is_not_treated_as_permission(veto_inputs):
    V.ROUTING_VETO.prediction_path(veto_inputs).unlink()
    with pytest.raises(V.HistoricalVetoBlocked, match='absent'):
        V.load_hard_vetoes(root=veto_inputs)
