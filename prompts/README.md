# Prompts

`example_prompts.txt` is a short inference list.
`MovieGenVideoBench.txt` is the MovieGen evaluation list.

Training uses `vidprom_filtered_extended.txt`, the Self-Forcing VidProm
split also used by RollingForcing. That file is about 140MB and is not
stored in git. Download it into this directory:

```bash
hf download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```

Keep the original file bytes and line order. Pair generation and on-policy
stages both read this list.
