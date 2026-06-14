I built a CLI Student Grade System using OOP classes, file I/O, and a tiebreaker leaderboard ranked by curved GPA.
I was surprised by how the curved GPA can actually punish weak students further when the class average is already high — the "fair" curve turned out to be unfair.
I don't fully understand how to handle a true three-way tie where curved GPA, raw GPA, and all tiebreakers are identical — the current code would just preserve CSV order, which feels arbitrary.

## Structure

```
├── day-01/          ← Python OOP — Student Grade System
│   ├── student.py                # Student class with OOP
│   ├── grader.py                 # CLI grader + leaderboard
│   ├── students.csv              # Sample student data
│   └── README.md

learning-journal/
└── day-01.md

```
