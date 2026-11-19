"""Leave-one-task-out text for the few-shot VLM baseline.

Each block is another task's annotated steps and the mistakes labeled in the
recordings. The task being scored is left out. Stapler has no labeled mistakes.
"""

from __future__ import annotations

TASKS = (
    {
        "id": "sim",
        "name": "Replace a SIM card",
        "steps": (
            "use the needle to open the SIM card slot",
            "put the SIM card into the SIM card slot",
            "press the SIM card slot back",
        ),
        "mistakes": (
            "Step 2, put the SIM card into the SIM card slot: the SIM card is inserted the wrong way.",
        ),
    },
    {
        "id": "smoothie",
        "name": "Make a strawberry smoothie",
        "steps": (
            "cut strawberries",
            "add strawberries to blender cup",
            "add sugar",
            "add milk to blender cup",
            "add ice cream to blender cup",
            "prepare blender for blending",
            "blend",
            "pour smoothie into serving glass",
            "spray cream onto smoothie",
        ),
        "mistakes": (
            "Step 1, cut strawberries: the strawberries are not cut.",
            "Step 3, add sugar: the sugar is skipped.",
            "Step 4, add milk to blender cup: chocolate milk is used instead of milk.",
        ),
    },
    {
        "id": "toilet",
        "name": "Clean a toilet",
        "steps": (
            "apply detergent to the interior of the toilet",
            "scrub the toilet interior",
            "flush and wash the interior",
        ),
        "mistakes": (
            "Step 1, apply detergent: water spray is used instead of detergent.",
            "Step 1, apply detergent: shaving cream is used instead of detergent.",
            "Step 1, apply detergent: the detergent is skipped.",
        ),
    },
    {
        "id": "battery",
        "name": "Replace a battery in a TV remote",
        "steps": (
            "open cover",
            "remove battery",
            "put battery in",
            "close cover",
        ),
        "mistakes": (
            "Step 3, put battery in: the battery is inserted the wrong way.",
            "Step 3, put battery in: only 1 cell is used.",
        ),
    },
    {
        "id": "cpr",
        "name": "Perform CPR",
        "steps": (
            "check the foreign matter in the mouth",
            "check the breath of the patient",
            "check the pulse of the patient",
            "press the chest of the patient",
        ),
        "mistakes": (
            "Step 2, check the breath of the patient: the breath check is missed.",
            "Step 3, check the pulse of the patient: the pulse check is done wrong.",
            "Step 3, check the pulse of the patient, and step 4, press the chest: "
            "the pulse check is missed, and the chest press is done on the stomach instead of the chest.",
        ),
    },
)


def example_text(held_out: str) -> str:
    """Steps and annotated mistakes for every task except `held_out`."""
    ids = {task["id"] for task in TASKS}
    if held_out not in ids:
        raise ValueError(f"Unknown task {held_out!r}")
    blocks = [
        "EXAMPLES",
        "These are other tasks, not the task in the videos. For each one, the steps "
        "are the required procedure and the mistakes are what was annotated in our "
        "recordings. Use them only as examples of mistakes. Do not treat their steps "
        "as steps of this task.",
    ]
    for task in TASKS:
        if task["id"] == held_out:
            continue
        steps = "\n".join(f"{index}. {label}" for index, label in enumerate(task["steps"], start=1))
        mistakes = "\n".join(f"- {line}" for line in task["mistakes"])
        blocks.append(f"{task['name']}\nSteps:\n{steps}\nAnnotated mistakes:\n{mistakes}")
    return "\n\n".join(blocks) + "\n"
