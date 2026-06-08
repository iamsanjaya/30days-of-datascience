# Day 05 — June 07, 2025

## What I built today

Full exploratory data analysis on the Titanic dataset using Pandas — answered 5 business questions (survival correlates, deck vs class patterns, family size effects, age ranges, and a heuristic scoring formula) with supporting visualizations. Also built an out-of-box analysis extracting name titles from the Name column to uncover a hidden social hierarchy that standard tutorials never touch.

## The out-of-box challenge result

Extracted titles (Mr., Mrs., Miss., Master., Rare) from the Name field using regex and found that `Master` boys survived at 57.5% — nearly matching adult women — while nobility titles like Countess and Sir offered zero survival advantage over standard 1st class. The Name column encodes gender, age, marital status, and social rank simultaneously, making it one of the most information-dense fields in the dataset. Most Titanic analyses never open it.

## What surprised me

Two things. First, the Age × Pclass cross-tab: 2nd class children survived at 100%, while 3rd class adults over 36 survived at just 8.6% — the same dataset, wildly different worlds. Second, the heuristic survival score formula (built from just 5 rules, no ML) cleanly separated passengers into tiers ranging from 11.5% to 93.4% survival. That a few human-readable rules could approximate the outcome this well made gradient descent from Day 4 feel more real — the model is just finding these weights automatically.

## What I don't fully understand yet

The missingness in Cabin (77.1%) bothers me. I used it to extract deck letters but only 204 passengers had cabin data — so the deck analysis is really a analysis of a very specific, non-random subset. Passengers without cabin records were almost certainly 3rd class, meaning the deck survival rates are partially confounded by class. I flagged this but don't yet know the right statistical way to handle or communicate that confound formally.

## GitHub commit made: ✅

`day-05: titanic EDA with original finding`

## Tomorrow's priority

Day 06 — Multi-Table Merge + Missingness Analysis (NYC TLC Dataset). The missingness question from today carries directly into tomorrow's out-of-box challenge.
