<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Troubleshooting

## How to approach an Ethos-U problem

Most Ethos-U issues fall into one of three groups:

- The target profile or backend is not available.
- The model does not map well to the target.
- The run succeeds, but the numbers are not telling a clear story yet.

Start by deciding which group you are in. That makes it much easier to pick the
right next check.

## Target or backend not available

### Target profile not found

If MLIA cannot find the target profile, start with discovery rather than with
performance debugging.

- Run `mlia-target list` to confirm the profile name.
- Use an explicit TOML path if you are testing a custom profile.
- Check that `mlia-ethos-u` is installed in the active environment.

If the profile is missing from discovery, there is no point debugging Vela or
Corstone yet.

### Vela or Corstone backend missing

If the backend is unavailable, confirm installation first.

- Run `mlia-backend list`.
- Install or reinstall the backend through the MLIA backend-management workflow
  in your environment.

## Model does not map cleanly

### Many unsupported operators

When Vela reports broad compatibility problems, treat that as a model-mapping
question before treating it as a performance question.

- Review whether the model is properly quantized.
- Check whether the model includes operators that do not map well to Ethos-U.
- Rerun with `--compatibility` to isolate support issues before reading
  performance output.

### Model format problems

If the model is rejected early or support looks much worse than expected, check
the input itself first.

- Ensure the input model is a valid `.tflite` file.
- Prefer quantized TFLite models for Ethos-U analysis.
- Re-check export settings if operator coverage looks unexpectedly poor.

## The run succeeds, but the numbers are not helpful yet

### Memory pressure looks high

If memory-related metrics dominate, investigate movement and layout before you
assume the arithmetic path is the main problem.

- Check `dram_access_cycles` and SRAM-related metrics.
- Compare the top-level cycle totals with operator-level hot spots.
- Try a different target profile or custom profile if that matches your
  hardware story.

### Corstone run is much slower than Vela

This is usually expected. Corstone is a heavier path aimed at deeper validation.

- Use Vela for rapid iteration.
- Reserve Corstone for deeper validation or when system context matters.

### Corstone metrics are hard to compare with Vela

That is normal too. The two backends are useful in different ways.

- Compare overall cycle-oriented metrics first.
- Then use Corstone's richer layer and bus-level signals to explain the
  difference.
- Treat differences as clues, not as proof that one backend is wrong.

## Escalation path

If you are unsure whether the issue belongs to the target package or the core
package:

1. Confirm the target profile and backend used in the run.
2. Check whether the failure happens before or after backend execution begins.
3. Move to the core `mlia` package only when the problem looks like CLI
   orchestration or plugin discovery rather than Ethos-U behaviour.
