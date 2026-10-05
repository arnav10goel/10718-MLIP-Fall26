# Video annotation guide

Reference format v1: one CSV per instructional video. Prepare it from the tutorial alone,
before reviewing execution videos or mistake labels.

## File location

Under your data root, save `annotations/reference/<task>/<video_id>.csv`.
Use the original task and video ID; keep original COIN annotations unchanged.
The default data root is the repository's ignored `data/` directory.

Start with [the smoothie template](../data/annotations/reference/MakeStrawberrySmoothie/-3C-VGhs2mo.csv)
and [its video](../data/videos/MakeStrawberrySmoothie/-3C-VGhs2mo.mp4).
These links work when the ignored data is available locally.

## Columns

Keep this header exactly:

```csv
step_id,action,start_s,end_s,requirements
```

| Column | Entry |
| --- | --- |
| `step_id` | Unique ID within this tutorial: `s001`, `s002`, etc. Keep existing IDs when editing. |
| `action` | One short, meaningful action, such as "cut strawberries" or "add milk". |
| `start_s` | Visible start, in seconds from the beginning of this exact video. |
| `end_s` | Visible end, in seconds. The interval includes its start and excludes its end. |
| `requirements` | Brief quantities or technique explicitly shown or stated in the tutorial; blank if none. |

## Labelling

- Use one row per action occurrence, in tutorial order. A repeated action gets a new ID.
- Separate meaningful actions, such as adding fruit, milk and sugar, then blending.
- Use numeric seconds, e.g. `12.5`, with `0 <= start_s < end_s <= video duration`.
  Adjacent actions can share a boundary. Mark the actual action, not an entire camera shot.
- Leave unclear intervals unlabelled. Gaps mean unknown; they do not establish correct behaviour.
- Record explicit requirements without inventing quantities or rules from recipe knowledge.
- Keep execution-step and mistake labels in separate evaluation files. This table describes the reference.

## Save and use

Open the CSV in Sheets or Excel, fill the rows, and export as UTF-8 comma-separated
CSV with decimal points. Keep all five columns, leave absent requirements empty,
and put explanatory text in `requirements` rather than extra header or footer rows.

The VLM runner accepts this reference CSV through `--checklist`, validates it
against the reference video's metadata duration, and passes the rows as JSON text.
The smoothie file contains a draft checklist; review it before use.

## Private execution labels

Save `annotations/execution/<task>/<video_id>.csv` under your data root. These
labels are for evaluation only: never pass them as a checklist or model input.
Use this separate header:

```csv
occurrence_id,reference_step_id,action,start_s,end_s,evidence_type,notes
```

| Column | Entry |
| --- | --- |
| `occurrence_id` | Unique occurrence ID in this execution: `e001`, `e002`, etc. |
| `reference_step_id` | Matching semantic step from the paired reference; blank if unmatched or unclear. |
| `action` | Short description of the observed procedural action. |
| `start_s`, `end_s` | Seconds in this execution, with start included and end excluded. |
| `evidence_type` | `visible_action` for observed performance; `result` for a result whose action is not shown. |
| `notes` | Visibility limits, uncertain interpretation or other reviewer notes. |

- Record execution order and timing, without forcing the reference sequence.
  Repeated steps get new occurrence IDs but can reuse a reference-step ID.
- Match purpose rather than equipment details. Leave unmatched actions blank;
  do not invent a reference step or renumber its IDs.
- Result intervals locate visible evidence, not when an unseen action occurred.
  Keep them separate from visible-action timing in evaluation.
- Leave unknown gaps unlabelled. A missing row does not establish a missed step
  or a correct execution. Error-event annotations require a separate format.
- Use the same numeric bounds and UTF-8 CSV export rules as the reference table.
  The reference loader does not accept this seven-column execution format;
  `load_execution_annotations` reads execution labels, and `score_current_steps`
  scores paired step IDs. Connecting execution intervals to prediction
  checkpoints is still pending.

The [second smoothie draft](../data/annotations/execution/MakeStrawberrySmoothie/2T1Et9xE5IQ.csv)
is paired with reference `-3C-VGhs2mo`. Its timestamps and interpretations need
human review. It is another edited tutorial, not a labelled mistake recording.
