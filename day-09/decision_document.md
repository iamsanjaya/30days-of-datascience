# Threshold Selection Decision Document

**For:** Hospital Administration / Clinical Review Board
**Subject:** Recommended Decision Threshold for AI-Assisted Disease Screening Model
**Prepared by:** Sanjaya Chaudhary

---

## The Problem in Plain Language

Our machine learning model reviews patient data and outputs a probability score between 0 and 1. A score above a chosen **threshold** triggers a flag: "recommend further testing."

The default threshold engineers use is **0.5** — meaning the model only flags a patient if it is more than 50% confident. This sounds reasonable. It is not appropriate for medical screening.

---

## Why 0.5 Is the Wrong Threshold for This Setting

When we use threshold = 0.5, our model makes **two kinds of mistakes**:

| Mistake            | What it means                             | Consequence                                |
| ------------------ | ----------------------------------------- | ------------------------------------------ |
| **False Negative** | Patient has disease; model says "healthy" | Patient goes home untreated                |
| **False Positive** | Patient is healthy; model says "disease"  | Patient gets an unnecessary follow-up test |

These are not equally harmful.

A false negative in a screening context means a sick patient walks away without treatment. Depending on disease progression, this delay can be irreversible. A false positive means a healthy patient undergoes an additional diagnostic test — inconvenient and mildly stressful, but rarely harmful.

---

## Our Recommendation: Threshold = 0.09

By lowering the threshold from 0.50 to **0.09**, we:

- **Eliminated missed cases** — the model catches all 28 positive cases in our 60-patient test cohort (zero false negatives)
- **Accepted an increase in false alarms** — 17 unnecessary referrals out of 60 patients screened
- **Reduced total weighted harm by 68%** (total cost dropped from 50 → 16, using a conservative estimate that a missed diagnosis is 10× more costly than a false referral)

This is not a data science decision. It is a clinical values decision. The numbers only tell us the **tradeoff**. The hospital decides how to weigh it.

---

## What Changes if the Cost Ratio Changes?

| If you believe a missed case is… | Recommended threshold |
| -------------------------------- | --------------------- |
| 2× worse than a false referral   | ~0.45                 |
| 5× worse                         | ~0.40                 |
| 10× worse (our recommendation)   | ~0.09                 |
| 20× worse                        | ~0.25                 |

The model does not change. The threshold does. **This is a policy choice, not a technical one.**

---

## What This Model Cannot Do

- It cannot replace physician judgment for borderline cases
- It was trained on one patient population — performance may differ for underrepresented groups
- It should be retested periodically as patient demographics and disease patterns shift

---

## Decision Required

The hospital administration should document the chosen cost ratio and the resulting threshold before deployment. This decision should be reviewed annually or whenever the patient population changes significantly.

_"A model that is 95% accurate but misses 40% of sick patients is not a good model for this hospital. Accuracy is not the right measure. Recall is."_
