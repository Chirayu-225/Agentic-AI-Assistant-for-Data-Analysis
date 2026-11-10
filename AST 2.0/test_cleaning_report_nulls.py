"""
test_cleaning_report_nulls.py — the cleaning report's AFTER null count ignores the cleaner's "<col>_original" side
columns (blank by design), while the column count still includes them and the BEFORE count is unchanged.

No LLM, no network. Run: python test_cleaning_report_nulls.py
"""
import sys

from profile_service import get_cleaning_report

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


def main():
    # amount: numeric with one unparseable value ("pending") and two real blanks. name: one real blank.
    rows = ["name,amount"]
    for i in range(1, 9):
        rows.append(f"n{i},{i * 10}")
    rows.append("n9,pending")
    rows.append("n10,")
    rows.append("n11,")
    rows.append(",5")
    csv = ("\n".join(rows) + "\n").encode()

    rep = get_cleaning_report(csv, "t.csv")
    check("report succeeds", rep.get("success") is True, str(rep.get("error")))
    cols = [s.get("name", s.get("column")) if isinstance(s, dict) else s for s in rep["schema"]]
    check("side column amount_original exists in the cleaned schema", "amount_original" in cols, str(cols))
    check("columns count includes the side column (2 -> 3)", (rep["before"]["columns"], rep["after"]["columns"]) == (2, 3),
          str(rep["before"]))

    # before: 2 blank amounts + 1 blank name = 3 (the text "pending" is not null yet)
    check("before.nullValues unchanged (3)", rep["before"]["nullValues"] == 3, str(rep["before"]))
    # after, non-side columns only: amount = 2 blanks + "pending" blanked = 3; name = 1 -> 4. Side column ignored.
    check("after.nullValues counts only non-side columns (4)", rep["after"]["nullValues"] == 4, str(rep["after"]))

    # cross-check against the cleaned preview: the side column really has blanks that would have been counted
    pv = rep["preview"]
    side_blank = sum(1 for r in pv if r.get("amount_original") in (None, "") or r.get("amount_original") != r.get("amount_original"))
    check("the side column is mostly blank (so counting it would have inflated the number)", side_blank >= len(pv) - 1)

    # a column the user's own file already had, named like a side column, is NOT ignored
    own = b"a,a_original\n1,\n2,\n3,x\n"
    rep2 = get_cleaning_report(own, "o.csv")
    check("a user's own 'a_original' column is still counted", rep2["after"]["nullValues"] >= 2, str(rep2["after"]))

    print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
