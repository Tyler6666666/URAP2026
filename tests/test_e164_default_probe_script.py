from pathlib import Path


def test_e164_default_probe_script_pins_cropdense_min050_configuration():
    script = Path("tools/run_e164_default_probe.ps1")

    assert script.exists()
    text = script.read_text(encoding="utf-8")

    assert "offline-selector-replay" in text
    assert "--reacquire-max-distance-px" in text
    assert "240" in text
    assert "--reacquire-appearance-weight" in text
    assert "0.50" in text
    assert "--reacquire-crop-weights" in text
    assert "crop_binary_dji_hardneg.pt" in text
    assert "--reacquire-crop-weight" in text
    assert "--reacquire-min-crop-drone-score" in text
    assert "--reacquire-global-reject-tracklet-rejected" in text
    assert "--reacquire-global-require-tracklet-confirmation" not in text
    assert "--reacquire-global-delayed-confirm-frames" in text
    assert "--reacquire-global-small-min-area" in text
    assert "--reacquire-global-small-max-distance-px" in text
    assert "--reacquire-global-small-memory-probation-frames" in text
    assert "48" in text
    assert "cropdense_min050_trackletveto_delay4_sizeadaptive_probation48" in text
