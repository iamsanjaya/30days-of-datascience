class Student:
    """Represents a student with a name and a list of grades."""

    def __init__(self, name: str, grades: list[float]):
        self.name = name
        self.grades = grades

    def gpa(self) -> str:
        """ " Return average grade, or 0.0 if no grades."""
        return round(sum(self.grades) / len(self.grades), 2) if self.grades else 0.0

    def grade_letter(self) -> str:
        """ " Assign letter grade based on GPA."""
        g = self.gpa()
        if g >= 90:
            return "A+"
        elif g >= 80:
            return "A"
        elif g >= 70:
            return "B+"
        elif g >= 60:
            return "B"
        elif g >= 50:
            return "C+"
        elif g >= 40:
            return "C"
        elif g >= 35:
            return "D"
        else:
            return "NG"

    def __repr__(self) -> str:
        return f"Student (name = {self.name!r}, gpa = {self.gpa()}, grade = {self.grade_letter()})"
