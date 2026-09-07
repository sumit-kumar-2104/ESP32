# Hardware validation checklist (schema only)

No hardware measurements exist in this branch. This file documents the
schema that any future physical validation MUST use so numbers are never
fabricated. Every measurement record is one row.

```yaml
measurement_id: "<uuid>"
measured_at: "<ISO 8601 with timezone>"
operator: "<name-or-handle>"
hardware:
  metasurface_serial: ""
  csi_receiver:
    model: "Intel 5300"
    antennas: 3
  driver_version: ""
  firmware_version: ""
software:
  git_commit: ""
  git_dirty: false
  script: ""
calibration:
  reference_signal: ""
  phase_calibration_dbm: null
  gain_calibration_db: null
  weights_active:
    quantized_bits: 2
    mode: "" # continuous | quantized | STE-only
inputs:
  gesture_sequence: []
  intended_symbols: []
predicted:
  accumulated_outputs_db: []
  classifier_predictions: []
measured:
  accumulated_outputs_db: []
  classifier_predictions: []
timing:
  end_to_end_latency_ms: null
  preprocessing_latency_ms: null
  readout_latency_ms: null
  relative_phase_drift_ns: null
energy:
  wall_power_mw: null
notes: |
  Optional free-form notes. Never populate any of the above with software
  simulation values. If a field is not measured, leave it null.
```

A measurement record with any field marked `null` may not be used to
support a scientific claim about that field. Aggregated results MUST
attach the list of contributing `measurement_id` values.
