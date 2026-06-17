# Model V5

Model V5 is the first explicit **LiDAR reliability-map** experiment.

## Goal

Instead of trying to pinpoint every faulty BEV cell, V5 predicts a regional unreliability heatmap:

```text
input:  faulty 14-channel LiDAR/radar BEV + known fault type + known severity
output: 1-channel LiDAR unreliability heatmap, shape 1 x 400 x 400
```

The output should be interpreted as repair priority:

```text
high value = trust LiDAR less / repair more
low value  = trust LiDAR more / preserve more
```

## Target Creation

The point-level clean-vs-faulty difference is still computed first:

```text
diff(x,y) = max_c abs(clean_BEV[c,x,y] - faulty_BEV[c,x,y])
pointwise_soft = clip(diff / target_threshold, 0, 1)
binary_target = diff > target_threshold
```

Then V5 converts the pointwise target into a regional heatmap:

```text
blurred = repeated 3x3 box blur(pointwise_soft)
regional_target = max(pointwise_soft, normalize(blurred))
```

The default is:

```text
--heatmap-blur-iterations 6
```

Increase this for broader repair regions. Decrease it for sharper masks.

## Initial Local Test

PowerShell:

```powershell
cd "C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest"

$PY = "C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest\.venv311\Scripts\python.exe"
& $PY model_v5\run_model_v5_initial_local.py
```

This defaults to:

```text
samples: 5000
epochs: 50
LiDAR aggregation: 3 frames
radar aggregation: 50 frames
batch size: 1
base channels: 32
output: C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\model_v5_initial_5k
```

To only generate samples:

```powershell
& $PY model_v5\run_model_v5_initial_local.py --skip-training
```

To resume/continue training manually:

```powershell
$OUT = "C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\model_v5_initial_5k"

& $PY model_v5\train_model_v5.py `
  --dataset-dir "$OUT\dataset" `
  --model-path "$OUT\models\model_v5_best.pt" `
  --latest-model-path "$OUT\models\model_v5_latest.pt" `
  --metrics-path "$OUT\models\model_v5_metrics.csv" `
  --resume-from "$OUT\models\model_v5_latest.pt" `
  --epochs 50 `
  --batch-size 1 `
  --loader-workers 0 `
  --channel-normalization dataset `
  --normalization-samples 2048 `
  --threshold 0.65 `
  --soft-weight 1.0 `
  --bce-weight 0.5 `
  --dice-weight 1.0 `
  --range-loss-weight 0.25 `
  --range-channel-index 4 `
  --base-channels 32 `
  --early-stop-patience 50
```
