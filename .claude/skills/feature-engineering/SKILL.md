---
name: feature-engineering
description: Methodology and checklist for both Stage 1 (EDA, missing-value imputation) and Stage 2 (feature engineering, variable editing, outlier analysis, dimensionality reduction) on the Nova Academy Dropped_Course project. Use when investigating a variable in train.ipynb, deciding how to handle missing values, deriving a new feature, encoding a categorical/text variable, deciding on outlier treatment, or reducing dimensionality on Train_Data.csv.
---

# EDA, imputation & feature engineering — Nova Academy project

Two stages, in order. Stage 2 rules assume Stage 1 already established,
per variable: is it dirty, what's its missingness mechanism, does it
correlate with `Dropped_Course`, does it correlate with other features.
Don't jump to encoding/derivation for a variable that hasn't had its
Stage 1 pass yet — check `train.ipynb` and `stage1_plan.md` first.

**Unit of a row: one row = one order (a group booking for a course),
not one individual.** `Professionals_Count`/`Students_Count`/
`Observers_Count` describe the *composition* of that group, not
separate rows. Keep this in mind for any feature involving group size
or ratios — e.g. `Students_Count / (Professionals_Count +
Students_Count + Observers_Count)` is a per-order composition ratio,
not a per-person feature.

## Project & variable glossary (official schema)

Client-provided definitions — use these over any inferred meaning.

| Column | Meaning |
|---|---|
| `Client_ID` | Unique identifier for each order. |
| `Professionals_Count` | Number of experienced professionals registered in the order. |
| `Students_Count` | Number of students registered in the order. |
| `Observers_Count` | Number of observers (do not perform practical exercises) registered in the order. |
| `Course_Start_Date` | Intended start date of the course. |
| `Practical_Hours` | Planned practical training hours for the course. |
| `Theory_Hours` | Planned theoretical learning hours for the course. |
| `Registration_Days_Before` | Days between registration date and course start date. |
| `Origin_Country` | Country code of the group that placed the order. |
| `Catering_Package` | Type of catering package pre-ordered for the group. |
| `Welcome_Gift_Type` | Type of welcome gift distributed to the group. |
| `Requested_Lab_Config` | Lab environment the group requested (Linux/Mac workstation, etc). |
| `Assigned_Lab_Config` | Lab environment actually allocated on the day of the course. |
| `Prev_Course_Dropouts` | Number of times **this group** canceled courses in the past — client/group-level history, not a running dataset-wide count. |
| `Prev_Course_Attended` | Number of times this group participated in courses under the company. |
| `Pre_Course_Supports_Tickets` | Number of support requests the group opened before the course started. |
| `Physical_Course_Kits` | Number of physical equipment kits prepared in advance for the group. |
| `Waiting_List_Days` | Days the group waited on the waiting list until registration was confirmed. |
| `Registration_Changes` | Number of times the group changed its order details in the system. |
| `Enrollment_Type` | Method of registration with the company. |
| `Lanyard_Color` | Badge lanyard color for the group. |
| `Client_Category` | Business segment the group belongs to (Fintech, SaaS, etc). |
| `Submission_Source` | Registration channel the order arrived through. |
| `Returning_Client` | 1 = returning client, 0 = new client. |
| `Agent_ID` | Sales agent/representative who closed the registration. |
| `Company_ID` | Identifier of the company through which the group registered — explains the 95.1% missingness: most orders aren't registered through a formal company channel. |
| `Payment_Terms` | Payment terms defined for the order. |
| `Daily_Tuition_Cost` | Daily tuition cost for the order. |
| `Dropped_Course` | Target. 1 = group canceled, 0 = did not. |

**`Test_Data_No_Target.csv`** — same schema minus `Dropped_Course`,
used for the final prediction/scoring. Not present in this repo yet,
but every Stage 1/2 decision must be re-runnable on it: any fitted
value (medians, category encodings, smoothing) must be **fit on train
only** and applied to test with the same rule — never fit or peek at
test data. Wrap imputation/encoding steps as functions/mappings
learned from `Train_Data.csv`, not inline one-off computations, so
they transfer cleanly.

## Stage 1 — EDA & missing-value imputation

### The four questions every variable's cell must answer

1. **Is it dirty?** Junk characters, sentinel values, wrong dtype —
   fix *before* computing any stats, or the stats are wrong. Verified
   examples from this dataset:
   - `Origin_Country` raw has 721 "unique" values; `str.upper().str.replace(r"[^A-Z]","",regex=True)` collapses it to 154 real codes (`" PRT#"` / `"prt"` / `"PRT"` are the same value).
   - Six more columns show the identical corruption pattern:
     `Catering_Package` (321→4), `Payment_Terms` (236→5),
     `Client_Category` (505→8), `Lanyard_Color` (240→5),
     `Enrollment_Type` (298→4), `Submission_Source` (328→6).
   - `Students_Count` has a `9999.0` sentinel mixed into an otherwise
     0–3 range — not a real headcount, recode to `NaN` before
     `describe()`/`mean()`/anything else touches it.
2. **What's the missingness rate, and does missingness correlate with
   the target?** This decides the imputation strategy — see the table
   below. Always check both sides:
   `df.groupby(df[col].isna())[TARGET].mean()`.
3. **Does the variable correlate with the target?** Report the effect
   directly from the groupby table/chart (per-group rate vs. baseline,
   with n per group) — do not add a Mann-Whitney U or chi-square test
   by default. Only add a formal test when the visual/tabular effect
   is genuinely ambiguous (bars close to baseline, or comparable groups
   with overlapping spread) and every group being compared has enough
   n to make a test meaningful in the first place. Verified examples
   where a test would have been redundant: `Prev_Course_Dropouts`
   (36.6%→97.4% across n=58,184/5,084), `Returning_Client` (41.9%→24.0%
   across n=61,742/1,722) — large, well-powered, visually obvious
   effects need no test to confirm them.

### Every number in an interpretation must come from the cell's own output

If a statistic is written into the markdown interpretation (a group
mean, a median, an aggregated rate), the code cell directly above it
must actually `print()` or plot that number — not a side computation
done outside the notebook (e.g. in a scratch script) and then typed
into the markdown from memory. If the cell doesn't show it, either add
the `print`/annotation to the cell first, or don't make the claim.
Two real examples that violated this and were fixed:
- `Registration_Days_Before`: interpretation claimed dropped-group vs.
  stayed-group medians (104 vs. 43 days) that only existed in a
  side computation — the cell only drew a boxplot. Fixed by adding
  `print(stayed.describe()[["mean","50%"]])` /
  `print(dropped.describe()[["mean","50%"]])` to the cell.
- `Prev_Course_Attended`: interpretation claimed a `0` vs. `1+`
  aggregate rate (42.1% vs. 7.4%) that the cell's per-exact-value
  `rate_by_val` table never computes. Fixed by adding an explicit
  `df.groupby(df[col] > 0)[TARGET].agg(["mean","count"])` print.

A reader should be able to verify every number in the write-up by
reading the cell's printed output alone, without re-deriving anything.

### Interpretation write-up style

Structure each interpretation cell as **one bullet per plot, in the
order the plots appear in the code cell above it**, each bullet
labeled with which plot it's reading from (e.g. "**Histogram —
right-skewed distribution.**"). Follow with a **Decision** paragraph
written in plain, first-person ("we") conversational sentences — not
a dense, telegraphic list of clauses. Explain *why* each decision
follows from the finding above it, the way you'd talk a colleague
through it out loud, not like a compressed spec line.

Verified example (`Registration_Days_Before`), the target shape:

> 1. **Histogram — right-skewed distribution.** Mean (102.9) sits well
>    above the median (65)... median is the trustworthy summary, not mean.
> 2. **Boxplot — stayed vs. dropped.** Dropped clients registered much
>    further in advance... a longer gap goes with a higher chance of dropping.
> 3. **Missingness bar chart.** Their drop rate sits right on the
>    baseline... missingness itself carries no signal (MCAR-like).
>
> **Decision:** we're keeping this one as a numeric predictor — the
> stayed-vs-dropped gap is real, not noise. When we describe it later,
> we'll use the median rather than the mean, since the skew makes the
> mean misleading. The 629-day outlier is worth a second look in Stage
> 2, but nothing to act on yet. And since missing values don't line up
> with the target at all, a plain median fill in Stage 3 will do — no
> need to treat it as its own category.

4. **Does it correlate with other features?** Feeds the Section 2
   correlation matrix (Spearman for numeric×numeric, Cramér's V for
   categorical×categorical) — also flags redundant features before
   Stage 2 doubles up on the same signal.

### Missing-value decision framework

| Missingness pattern | Verified example | What to do |
|---|---|---|
| ≈ target baseline, no strong correlate | `Registration_Days_Before`: missing-rate target 41.5% vs. non-missing 41.4%, looks MCAR | Simple imputation is defensible — median for skewed numerics, mode/"UNKNOWN" for categoricals. |
| Missingness itself predicts the target | `Company_ID`: 95.1% missing; present-rate drop 21.2% vs. missing-rate 42.5% | Don't erase the signal by filling it. Keep a `_missing`/`has_x` indicator, or leave an explicit "MISSING" category (as done for `Origin_Country`). |
| Near-total missingness | `Company_ID` (95.1%), `Agent_ID` (17.6%) | Filling the value itself is close to meaningless at this rate; the presence/absence flag is where the real information is. |
| Sentinel value disguised as data | `Students_Count = 9999` | Not "missing" — mislabeled. Recode to `NaN` first, then treat like any other numeric missingness. |

### Before/after distribution comparisons

Required only for columns where you actually fill values (not for
ones left as an explicit "MISSING" category or turned into a flag) —
plot the distribution pre- and post-imputation side by side, so the
imputation choice is visually justified, not just asserted.

### Stage 1 is observational

No permanent drops, no derived columns, no encoding during this pass
— just cleaned raw data, the four-question findings, and one written
interpretation per variable. Save the "what do we do about it"
decisions for Stage 2.

## Stage 2 — feature engineering & variable editing

### The non-negotiable rule for every new feature

Every derived feature gets three things, written down (in the
notebook markdown cell next to it, not just said in chat):
1. **The logic** — what real-world mechanism it's supposed to capture.
2. **The expected effect on the target** — direction, and why.
3. **The math**, if any transform beyond a raw column rename is
   involved — formula + a one-line explanation of each term.

A feature with no stated mechanism is a fishing expedition, not
engineering. If you can't say *why* it should predict
`Dropped_Course` before you compute it, don't add it.

### Encoding decision tree (by cardinality + missingness)

Don't default to one-hot. Check cardinality and missingness first —
this dataset already showed both extremes:

| Situation | Example (verified) | What to do |
|---|---|---|
| Low cardinality (≤10), clean | `Requested_Lab_Config` (8), `Welcome_Gift_Type` (4) | One-hot. Cheap, no leakage risk. |
| High cardinality, dirty text | `Origin_Country` raw: 721 values → 154 after cleaning | Clean first (see Stage 1), *then* re-evaluate cardinality. Cleaning alone can drop cardinality 4–5x. |
| High cardinality after cleaning (>~20 categories) | `country_clean` still has 154 categories, `PRT` alone = 42% of rows | **Smoothed target encoding**, not one-hot (one-hot here = 154 sparse columns = curse-of-dimensionality territory for 63k rows). See formula below. |
| Missingness itself predicts the target | `Company_ID` (see Stage 1 table) | Encode presence as its own binary flag (`has_company_id`), separately from whatever you do with the ID values that do exist. |
| Sentinel value disguised as data | `Students_Count = 9999` | Recode sentinel → `NaN` in Stage 1, before any Stage 2 encoding or outlier logic touches the column. |
| High-cardinality entity ID with real gaps | `Agent_ID` (203 unique, 17.6% missing) — the specific salesperson who closed the order | Same treatment as `Company_ID`: don't one-hot 203 raw IDs. Check missingness-vs-target first; if the present values show real spread, use smoothed target/frequency encoding (agent track record) plus a `has_agent_id` flag, not the raw ID. |

#### Smoothed target encoding — the formula, and why plain target-rate isn't enough

Plain per-category mean (`groupby(col)[target].mean()`) overfits categories
with few rows — a country with n=1 and one dropout shows a 100% drop
rate that's pure noise, not signal. Shrink small-n categories toward
the global mean:

```
smoothed = (n_cat * mean_cat + k * mean_global) / (n_cat + k)
```

- `n_cat` — rows in that category
- `mean_cat` — raw target rate for that category
- `mean_global` — dataset-wide target rate (0.4144 for `Dropped_Course`)
- `k` — smoothing strength (pseudo-count of "prior" global-mean rows);
  higher `k` pulls small categories harder toward the global mean

Verified on this dataset with `k=20`:
- `PRT` (n=26,429, raw mean 0.6378) → smoothed **0.6376** — barely
  moves, because n is huge relative to k.
- `AGO` (n=309, raw mean 0.5761) → smoothed **0.5662** — pulled
  noticeably toward baseline.
- A singleton country like `ABW` (n=1, raw mean 0.0) → smoothed
  **0.3947** — almost entirely replaced by the global mean (0.4144),
  which is exactly the point: one row shouldn't get to claim a 0%
  drop rate.

Compute smoothed encodings on a train-fold only if you're doing CV —
fitting the encoding on the full dataset including validation rows
leaks target information.

### Curse of dimensionality

- Count your effective feature count *before* modeling: raw column
  count + (one-hot columns added) − (columns replaced by target/
  frequency encoding). Six of this dataset's raw categoricals look
  like 200–500-category monsters raw, but collapse to 4–8 real
  categories once cleaned — always clean before deciding an encoding
  strategy, the "high cardinality" problem is often fake.
- For genuinely high-cardinality columns after cleaning (`country_clean`
  at 154), prefer target/frequency encoding over one-hot — one-hot
  would add 150+ mostly-empty columns for 63k rows, inflating the
  feature space without adding information density.
- If two features are near-duplicates in what they capture (e.g. a
  `Requested_Lab_Config`/`Assigned_Lab_Config` mismatch flag would
  likely correlate with `Waiting_List_Days`), check the Stage 1
  cross-variable correlation section before adding both — redundant
  correlated features add dimensionality without adding signal.
- `Requested_Lab_Config` vs `Assigned_Lab_Config`: per the glossary,
  these are "what the group asked for" vs. "what they actually got on
  the day" — a `lab_config_mismatch` binary flag (unmet request) is a
  legitimate, well-motivated feature, not a guess. Confirm it against
  the target before keeping it.

### Outlier analysis

- Visualize before deciding — boxplot or histogram, same pattern as
  the Stage 1 numeric cells in `train.ipynb` (`plt.boxplot([stayed,
  dropped], ...)`).
- Distinguish a **sentinel value** (`Students_Count = 9999`, a
  data-entry code, not a real outlier — handled in Stage 1) from a
  **genuine extreme** (e.g. `Waiting_List_Days` max = 391, a real long
  wait, not corrupted data — decide whether it's a legitimate tail or
  worth capping based on how many rows are affected and whether they
  cluster on one target class).
- State the rule and the row count it affects — "dropped N rows where
  X > threshold, Y% of dataset, drop rate among them was Z" — not just
  "removed outliers."

## Reference

`stage1_plan.md` — full variable priority order and what's covered vs.
pending. `train.ipynb` — the actual EDA/imputation notebook this
skill's examples are drawn from.
