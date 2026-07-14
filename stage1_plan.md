# Nova Academy — Stage 1 (EDA + Imputation) Plan

## Context

This is Stage 1 of a 2-stage B2B course-cancellation prediction project (`Train_Data.csv`, 63,464 rows × 29 cols, target `Dropped_Course`). Stage 1 is **observational only** — no feature engineering, no outlier removal, no modeling. Deliverable is a notebook that is explicit, visual, and human-narrated: every variable gets its own cell, a chart, a stat test, and a short written interpretation prompt. Stage 2 (later) will build features/reduce dimensionality on top of what Stage 1 concludes — so Stage 1's interpretations are load-bearing, not decorative.

Already confirmed by direct inspection of the CSV:
- **6 categorical columns are systematically dirty** — not just `Origin_Country`. `Catering_Package` (321→4), `Payment_Terms` (236→5), `Client_Category` (505→8), `Lanyard_Color` (240→5), `Enrollment_Type` (298→4), `Submission_Source` (328→6) all collapse to a handful of true categories once junk chars/casing are stripped (same corruption pattern as `Origin_Country`: stray symbols, whitespace, case variants). This means text cleaning must be a **reusable first step**, applied uniformly, not hand-rolled per column.
- `Requested_Lab_Config`/`Assigned_Lab_Config`/`Welcome_Gift_Type` are already clean (small nunique, no junk).
- **`Students_Count` has a `9999.0` sentinel value** mixed into an otherwise 0–3 range — a fake-missing/outlier flag, not a real count. Other numeric columns need the same sentinel scan before being treated as clean.
- `Company_ID` is 95.1% missing, `Agent_ID` 17.6% missing — likely structurally missing (e.g. individual vs. corporate client), not random gaps.
- `Course_Start_Date` is a real date column (666 unique dates) — not yet parsed to datetime.
- `Client_ID` has zero duplicates — confirmed row-level identifier, excluded from analysis.

## Notebook structure (`train.ipynb`)

**Section 0 — Setup**
- Load CSV, set `TARGET`, print shape + baseline drop rate.
- One `clean_text(series)` helper (uppercase, strip non-alphanumeric, collapse whitespace) — reused by every dirty categorical column, not reimplemented per cell.
- Three investigate helpers, one per variable type, each producing: summary stats, one primary chart, and a printed prompt block ("What can you infer about `<col>`?") for the human interpretation to be written directly underneath in a markdown cell. Report the effect straight from the groupby table/chart (rate vs. baseline, n per group) — only add a formal test (Mann-Whitney, chi-square) when the effect is genuinely ambiguous from the chart and every group compared has enough n to make a test meaningful:
  - `investigate_numeric(col)` — describe(), histogram + boxplot, sentinel/outlier flags, per-value/per-bucket rate vs. target.
  - `investigate_categorical(col, clean=True)` — value_counts, bar chart of category rate vs. target baseline, rare-category summary.
  - `investigate_datetime(col)` — parse, distribution over time, rate-over-time line chart, seasonality check vs. target.
- Every helper returns the summary table (not just prints) so it can feed the correlation section later.
- Every number cited in an interpretation markdown cell must be printed or plotted by the code cell directly above it — no side computation done outside the notebook and typed in from memory. If a stat is needed for the write-up, add the `print`/annotation to the cell first (see `feature-engineering` skill for two real examples this caught).

**Section 1 — Variable-by-variable pass, in priority order (below)**
Each variable gets: 1 code cell (`investigate_x("Column_Name")`) + 1 markdown cell (your written interpretation, prompted by the printed question). No column skipped, low-priority ones just come last.

**Section 2 — Cross-variable correlation**
- Numeric×numeric: correlation heatmap (Spearman, since several numerics are count-like/skewed).
- Categorical×categorical: Cramér's V matrix for the cleaned categoricals.
- Full feature×target summary table (effect size + p-value per variable), assembled from the return values of Section 1 — this is the artifact that drives imputation decisions and previews Stage 2 priorities.

**Section 3 — Missing value imputation**
- Decided per-column *from Section 1/2 findings*, not upfront guesses (e.g. missingness-rate-vs-target-rate comparison decides MCAR-style mean/mode fill vs. "own category" fill vs. flag-and-impute).
- Explicit before/after distribution plot for every column where imputation visibly shifts the shape.

## Priority queue (most → least important to examine)

**Tier 1 — direct behavioral signal, likely strongest predictors**
1. `Prev_Course_Dropouts` — numeric, prior cancellation history
2. `Returning_Client` — binary, repeat vs. new
3. `Prev_Course_Attended` — numeric, engagement history
4. `Registration_Days_Before` — numeric, lead time (4.2% missing)
5. `Waiting_List_Days` — numeric, friction signal
6. `Registration_Changes` — numeric, instability signal
7. `Pre_Course_Supports_Tickets` — numeric, friction/support load

**Tier 2 — dirty categoricals needing the clean_text pass, business-relevant**
8. `Origin_Country` (already prototyped)
9. `Client_Category`
10. `Payment_Terms`
11. `Enrollment_Type`
12. `Submission_Source`
13. `Catering_Package`
14. `Lanyard_Color`

**Tier 3 — course logistics / structure**
15. `Requested_Lab_Config` vs `Assigned_Lab_Config` — examine as a pair (mismatch = unmet request, plausible cancellation driver)
16. `Practical_Hours`, `Theory_Hours`
17. `Professionals_Count`, `Students_Count` (sentinel `9999` to resolve first), `Observers_Count`
18. `Physical_Course_Kits`
19. `Daily_Tuition_Cost`
20. `Course_Start_Date` — parse + seasonality

**Tier 4 — sparse / structurally-missing IDs**
21. `Agent_ID` (17.6% missing)
22. `Company_ID` (95.1% missing — likely individual-vs-corporate flag, examine missingness itself as the feature)
23. `Welcome_Gift_Type` (clean, low cardinality, likely weak signal)

**Excluded from variable-by-variable analysis**
- `Client_ID` — pure identifier, zero duplicates, no analytical value.

## Verification
- Run notebook top to bottom; every code cell must execute without error and produce a visible chart + printed stat block.
- Spot-check 2–3 of the "dirty" categorical cleanups (e.g. `Client_Category`, `Payment_Terms`) against raw `value_counts()` to confirm the `clean_text` collapse is correct, same way `Origin_Country` was verified.
