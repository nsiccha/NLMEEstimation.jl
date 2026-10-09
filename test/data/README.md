# Test data

- `theoph.csv`: R's `datasets::Theoph` (theophylline after a single oral dose in 12
  subjects; Boeckmann, Sheiner and Beal, *NONMEM Users Guide*, 1994), exported with
  R 4.5.2. `Dose` is in mg/kg, so the dose amount is `Dose * Wt` mg; `conc` is in mg/L
  and `Time` in hours.
