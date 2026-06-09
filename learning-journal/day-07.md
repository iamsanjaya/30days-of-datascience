# Day 07 — June 09, 2025

## What I built today

Six-chart EDA visual report on the NYC TLC Green Taxi cleaned dataset, each chart structured as an argument rather than a display. Every title is a one-line conclusion that could stand alone as a finding. Separately built a deception study: the same hourly demand chart rendered twice — once with a truncated Y-axis to manufacture apparent drama, once honest with zero baseline — with the deception mechanism documented in a caption.

## The out-of-box challenge result

The truncated-axis version made a ~15% demand variation read as a ~38% surge — more than double the apparent effect size. The deception required no data manipulation at all: same dataset, same chart type, one parameter changed (`ylim` start). This is precisely why it is so dangerous and so common. The exercise also forced me to quantify the distortion: `apparent_pct = (true_max - true_min) / (true_max - y_lie_min)` — a formula worth remembering.

## What surprised me

The tip chart (Chart 04) required filtering to `payment_type == 1` (card only) before the boxplot was meaningful. TLC does not record cash tip amounts at all — they appear as zero in the data, not as missing. Plotting unfiltered would have shown a deceptively low median tip across all days. This is the kind of domain-specific data trap that only reveals itself when you read the data dictionary carefully.The deception chart produced an unexpected result: true variation = 1129% — the peak hour has roughly 12x more trips than the quietest hour (3–4 AM). Normally a truncated axis exaggerates variation, but here the true swing is so large that the truncated version (99% apparent) actually understates it. This is a rare case where the honest chart is more dramatic than the misleading one.

## What I don't fully understand yet

The OLS trend line in Chart 02 (fare vs distance) fits well for the linear portion but the scatter above ~15 miles has high variance that a single straight line does not capture. I suspect this is because long trips include airport runs with flat-rate pricing, which breaks the linear assumption. I would need to segment by trip type to model this properly — something for the regression days ahead.

## GitHub commit made: ✅

`day-07: visualization storytelling + chart deception study`

## Tomorrow's priority: Day 08 — Linear Regression from scratch (Normal Equation + gradient descent)
