# %%
import csv
from student import Student


def load_students(filepath: str) -> list[Student]:
    """Load students from csv. Handles missing file, bad encoding, malformed rows."""
    students = []
    try:
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=2):  # line 1 = header
                try:
                    name = row["name"].strip()
                    grades = [float(g.strip()) for g in row["grades"].split(",")]
                    students.append(Student(name, grades))
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
    except UnicodeDecodeError:
        print("Encoding error - file must be UTF-8")
    return students


def apply_curve(students: list[Student], target_avg: float = 75.0) -> dict[str, float]:
    """
    Curve grades so the class average becomes target_avg.
    New grade = old_grade + (target_avg - class_Avg)
    Capped at 100.
    """
    if not students:
        return {}
    class_avg = sum(s.gpa() for s in students) / len(students)
    shift = target_avg - class_avg
    return {s.name: min(round(s.gpa() + shift, 2), 100.0) for s in students}


def tiebreaker_rank(
    students: list[Student], curved_gpas: dict[str, float]
) -> list[Student]:
    """
    Sort by curved GPA descending.
    Tiebreaker: higher raw GPA wins (more consistent performer).
    Design decision: raw GPA reflects true ability before artificial scaling-
    a student who earned 90 raw beats onw who needed a curve to reach 90.
    """
    return sorted(
        students,
        key=lambda s: (curved_gpas[s.name], s.gpa()),
        reverse=True,
    )


def print_leaderboard(students: list[Student], curved_gpas: dict[str, float]):
    """
    Print a formatted ranked leaderboard.
    """
    ranked = tiebreaker_rank(students, curved_gpas)
    print(f"\n{'Rank' : <6}{'Name' : <15}{'Raw GPA' :<12}{'Curved GPA' :<14}{'Grade'}")
    print("-" * 55)
    for i, s in enumerate(ranked, start=1):
        print(
            f"{i:<6}{s.name:<15}{s.gpa():<12}{curved_gpas[s.name]:<14}{s.grade_letter()}"
        )


# %%
if __name__ == "__main__":
    filepath = "students.csv"
    students = load_students(filepath)

    if students:
        curved = apply_curve(students, target_avg=75.0)
        print_leaderboard(students, curved)
    else:
        print("No valid student data to display.")
