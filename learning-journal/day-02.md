# Day 2 — June 4, 2025

## What I built today

Word frequency counter with top-20 bar chart, and a primitive sentiment analyzer
that plots the emotional arc across 62 chapters of Pride and Prejudice using
only manually curated word lists — no ML at all.

## The out-of-box challenge result

Built a scoring system with positive, negative, and tension word lists.
The arc correctly identified Chapter 33 as most positive (Darcy's proposal),
Chapter 47 as most negative (Lydia's elopement), and Chapter 39 as highest
tension — coherent with the actual novel structure.

## What surprised me

Smoothing the data for the plot shifted the peak detection results. Chapter 5
had the highest raw positive score but after rolling average, Chapter 4 ranked
higher because it averaged with Chapter 5's strong neighbours. Smoothing changed
the answer, not just the appearance.

## What I don't fully understand yet

Why the rolling window size affects peak shift differently depending on how
clustered the high-scoring chapters are. Need to think about this more with NumPy.

## GitHub commit made: ✅

`Day 2 — Word Frequency & Primitive Sentiment Arc`

## Tomorrow's priority: Day 3 — NumPy distance metrics from scratch
