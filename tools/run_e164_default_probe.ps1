param(
    [string]$Predictions = "C:\Users\pc\Desktop\tiny object detection\runs\ura27_sequence_gate_smoke\dji_broad_seg01_v2\predictions.jsonl",
    [string]$Video = "D:\datasets\my_video\validation_segments\dji_fly_20260522_113924_5x20s\raw_videos\dji_fly_20260522_113924_10_1779475848691_hdrvideo_seg01_000000_000598.mp4",
    [string]$CropWeights = "C:\Users\pc\Desktop\tiny object detection\runs\dji_dense_stage_b_hardneg_repair_20260530\models\crop_binary_dji_hardneg.pt",
    [string]$Out = "runs\e164_selector_probe\dji_broad_seg01_v2_e164_default_cropdense_min050_trackletveto_delay4_sizeadaptive_probation48",
    [switch]$SaveVideo
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

foreach ($PathToCheck in @($Predictions, $Video, $CropWeights)) {
    if (-not (Test-Path -LiteralPath $PathToCheck)) {
        throw "Missing required path: $PathToCheck"
    }
}

$ArgsList = @(
    "-m", "qstr_dronedet.cli", "offline-selector-replay",
    "--predictions", $Predictions,
    "--out", $Out,
    "--top-k", "5",
    "--max-jump-px", "48",
    "--max-recover-frames", "3",
    "--min-accept-score", "0.25",
    "--min-motion-consistency", "0.15",
    "--min-memory-consistency", "0.15",
    "--reacquire-max-distance-px", "240",
    "--reacquire-confirm-frames", "2",
    "--reacquire-min-motion-consistency", "0.20",
    "--reacquire-min-memory-consistency", "0.20",
    "--reacquire-stale-after-frames", "10",
    "--reacquire-global-min-area", "1000",
    "--reacquire-global-small-min-area", "100",
    "--reacquire-global-small-max-distance-px", "1200",
    "--reacquire-global-small-memory-probation-frames", "48",
    "--reacquire-global-max-area", "6000",
    "--reacquire-global-min-detector-score", "0.32",
    "--reacquire-global-reject-tracklet-rejected",
    "--reacquire-global-delayed-confirm-frames", "4",
    "--reacquire-appearance-weight", "0.50",
    "--reacquire-crop-weights", $CropWeights,
    "--reacquire-crop-weight", "1.0",
    "--reacquire-min-crop-drone-score", "0.50",
    "--video", $Video
)

if ($SaveVideo) {
    $ArgsList += "--save-video"
}

Write-Host "=== E164 default probe: cropdense_min050_trackletveto_delay4_sizeadaptive_probation48 ==="
Write-Host "Predictions: $Predictions"
Write-Host "Video:       $Video"
Write-Host "CropWeights: $CropWeights"
Write-Host "Out:         $Out"

python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "E164 default probe failed with exit code $LASTEXITCODE"
}
