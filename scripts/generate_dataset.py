#!/usr/bin/env python3
"""Create deterministic, fully synthetic train/validation/test JSONL splits."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SYSTEM = (
    "Create a concise ESKD care handoff. Use exactly these headings: "
    "Documented observations; Care or access items; Information to confirm; "
    "Human-review note. Use only facts in the notes. Do not diagnose or recommend treatment."
)

SCENARIOS = [
    (
        ["Transportation did not arrive for the scheduled visit.", "Patient reported feeling tired; no symptom details were documented.", "Family contact requested a callback about the missed visit."],
        ["The patient reported feeling tired; no additional symptom details were documented."],
        ["Transportation did not arrive for a scheduled visit.", "A family contact requested a callback about the missed visit."],
        "Confirm whether the missed visit was rescheduled and whether additional symptoms were reported.",
        "Please review the missed visit and follow up with the family contact.",
    ),
    (
        ["Patient asked to change an appointment time.", "Scheduling change request was received; no new time was documented.", "No response from the scheduling office was recorded."],
        ["The patient asked to change an appointment time.", "A change request was received, but no new appointment time was documented."],
        ["Scheduling follow-up is pending."],
        "Confirm the updated appointment time and whether it was communicated to the patient.",
        "Please review the unresolved scheduling request.",
    ),
    (
        ["Patient asked for an interpreter at the next visit.", "Interpreter request was submitted.", "Interpreter availability was not confirmed."],
        ["The patient requested an interpreter for the next visit.", "An interpreter request was submitted."],
        ["Interpreter availability remains unconfirmed."],
        "Confirm interpreter availability and communicate the result before the next visit.",
        "Please review and resolve the pending interpreter request.",
    ),
    (
        ["Voicemail was left after a scheduled check-in.", "Family contact reported that no voicemail was received.", "No alternate phone number was documented."],
        ["A voicemail was documented after the scheduled check-in.", "A family contact reported that no voicemail was received."],
        ["The scheduled check-in requires follow-up because contact was not confirmed."],
        "Confirm the preferred contact number and whether another outreach attempt was completed.",
        "Please review the conflicting contact information and coordinate another outreach attempt.",
    ),
    (
        ["Patient stated they were unsure which location to attend.", "Location confirmation message was sent.", "No acknowledgment of the location message was documented."],
        ["The patient stated they were unsure which location to attend.", "A location confirmation message was sent."],
        ["Attendance access may remain unclear because receipt of the location message was not documented."],
        "Confirm that the patient received and understood the location information.",
        "Please follow up on location confirmation before the scheduled visit.",
    ),
    (
        ["Patient reported that the written schedule was difficult to read.", "A clearer copy was offered.", "No record that the clearer copy was sent."],
        ["The patient reported difficulty reading the written schedule.", "A clearer copy was offered."],
        ["The status of the replacement schedule is not documented."],
        "Confirm whether a clearer schedule was sent and received.",
        "Please review the schedule communication before the next visit.",
    ),
    (
        ["Visit completed.", "Patient stated that transportation for next week is arranged.", "No callback request was documented."],
        ["A visit was completed.", "The patient stated that transportation for the following week is arranged."],
        ["No additional access concern was documented."],
        "Confirm transportation arrangements at the next contact.",
        "Please complete routine review of the next scheduled contact.",
    ),
]

TEST_NOTE_REWRITES = {
    "Transportation did not arrive for the scheduled visit.": "The scheduled ride was not available, and the appointment was missed.",
    "Patient reported feeling tired; no symptom details were documented.": "The nursing entry recorded fatigue but did not include further symptom detail.",
    "Family contact requested a callback about the missed visit.": "A family member asked to be contacted about the appointment that was not attended.",
    "Patient asked to change an appointment time.": "The patient requested a different appointment time.",
    "Scheduling change request was received; no new time was documented.": "Scheduling received the request, but the replacement time was not recorded.",
    "No response from the scheduling office was recorded.": "The record does not show a response from scheduling.",
    "Patient asked for an interpreter at the next visit.": "The patient requested language interpretation for the next appointment.",
    "Interpreter request was submitted.": "A request for an interpreter was entered.",
    "Interpreter availability was not confirmed.": "The record does not confirm that an interpreter is available.",
    "Voicemail was left after a scheduled check-in.": "Staff documented leaving a message after the planned check-in.",
    "Family contact reported that no voicemail was received.": "A family member said the message was not received.",
    "No alternate phone number was documented.": "No second phone number appears in the record.",
    "Patient stated they were unsure which location to attend.": "The patient said they did not know which site to go to.",
    "Location confirmation message was sent.": "Staff sent a message with the visit location.",
    "No acknowledgment of the location message was documented.": "There is no record that the location message was acknowledged.",
    "Patient reported that the written schedule was difficult to read.": "The patient said the printed schedule was not easy to understand.",
    "A clearer copy was offered.": "Staff offered to provide a more readable schedule.",
    "No record that the clearer copy was sent.": "The record does not show that a replacement schedule was sent.",
    "Visit completed.": "The scheduled visit was completed.",
    "Patient stated that transportation for next week is arranged.": "The patient reported that next week's ride has been arranged.",
    "No callback request was documented.": "No request for a return call appears in the notes.",
}


def handoff(observations: list[str], access: list[str], confirm: str, review: str) -> str:
    return "\n\n".join(
        [
            "Documented observations\n" + "\n".join(f"- {item}" for item in observations),
            "Care or access items\n" + "\n".join(f"- {item}" for item in access),
            f"Information to confirm\n- {confirm}",
            f"Human-review note\n- {review}",
        ]
    )


def make_example(split: str, index: int) -> dict:
    notes, observations, access, confirm, review = SCENARIOS[index % len(SCENARIOS)]
    if split == "test":
        notes = [TEST_NOTE_REWRITES[note] for note in notes]
    start = date(2026, 1, 1) + timedelta(days=index * 3)
    roles = ("Care coordinator", "Nurse", "Family contact")
    timeline = "\n".join(
        f"{start + timedelta(days=offset)} | {roles[offset]}: {note}"
        for offset, note in enumerate(notes)
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Synthetic case E-{split.upper()}-{index + 1:03d}\n{timeline}"},
            {"role": "assistant", "content": handoff(observations, access, confirm, review)},
        ]
    }


def write_split(name: str, count: int, offset: int) -> None:
    path = DATA / f"{name}.jsonl"
    rows = [make_example(name, offset + index) for index in range(count)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    write_split("train", 36, 0)
    write_split("valid", 8, 100)
    write_split("test", 12, 200)


if __name__ == "__main__":
    main()
