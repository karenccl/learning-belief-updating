"""Optional real-data extension.

Decision recorded on Day 1 (docs/notes/day1.md): no reversal-learning dataset was
confirmed against all eight feasibility criteria by the end of Day 1, so the
real-data analysis is omitted and the project is simulation-based by design.

If a dataset is confirmed later, this script should: keep a raw untouched copy in
data/optional_real_data/raw/, write a data dictionary, document exclusions before
fitting, validate 0/1 coding of choices and rewards, and then reuse
src.cross_validation.cross_validate and src.predictive_checks exactly as in the
simulations (including the perseveration benchmark; Katahira, 2018).
"""

if __name__ == "__main__":
    print("Real-data extension omitted: no dataset passed the Day 1 feasibility checklist.")
