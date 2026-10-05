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

The offline VLM runner accepts this reference CSV through `--checklist`, validates it
against the reference video's metadata duration, and passes the rows as JSON text.
The smoothie file contains a draft checklist; review it before use.
