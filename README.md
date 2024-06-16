# MMIM
Follow the steps contained in ``MMIM/README` to get the correct data.

Then inside the MMIM directory run the following commands:
```bash
bash src/train_baselines.sh
bash src/eval_baselines_missing.sh
bash src/extract_encoders.sh
bash src/train_cmams.sh
bash src/test_cmams_missing.sh
```

Read through the scripts to figure out the arguments and what each script does/where it saves results.