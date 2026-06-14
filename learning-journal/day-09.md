# Day 09 — June 11, 2025

## What I built today:

Logistic Regression pipeline with full diagnostic plots (ROC, confusion matrix, precision-recall). Implemented a threshold sweep showing how precision, recall, and F1 change across all thresholds. Built a cost-based threshold selection framework where false negatives carry 10× the weight of false positives, shifting the optimal threshold from 0.50 → 0.09, reducing total weighted cost by 34 units (from 50 → 16). Best F1 threshold: 0.33.

## The out-of-box challenge result:

Wrote a 1-page decision document for hospital administration explaining that threshold selection is not a technical decision — it is a clinical values decision. The model stays the same; the threshold encodes what the institution believes about the relative cost of different errors. Total weighted harm dropped ~90% by moving from default 0.5 to cost-optimal 0.35. False negatives went from 1 to 0; false positives increased from 1 to 2 — a clearly acceptable tradeoff in screening.

## What surprised me:

The Cleveland dataset's target column has values 0–4, not binary — any value above 0 means disease present. Also the CSV had a header row with column named "condition" not "target", which required updating config.py rather than hardcoding the name.

## What I don't fully understand yet:

How to set the cost ratio rigorously in practice. In this document I used 10:1 as an assumption — but in a real hospital setting, how do clinicians actually quantify the cost of a missed diagnosis vs an unnecessary referral? That seems like a health economics problem, not an ML problem. Worth reading more on QALY-based analysis.

## GitHub commit made: ✅

`Day 09 — Logistic Regression + Threshold as a Medical Ethics Decision`

## Tomorrow's priority: Day 10 — Decision Trees, visualize the tree, manual path tracing

## Note:

Committed on day, pushed 2026-06-12.
