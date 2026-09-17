# Working notes — NOT part of the manuscript

Submission target: Scientific Journal of Computer Science (SJCS).
File to submit: `paper/manuscript_SJCS.docx` (copy on Desktop as `SJCS_SickleCell_Manuscript.docx`).

## Before submitting
- **[repository URL]** placeholder in DATA AND CODE AVAILABILITY — create a public repo (code only; the dataset licence forbids redistributing derived crops) and paste the link.
- **Plagiarism report**: journal caps total similarity at 25%, each source at 4%. Not run — needs Turnitin/iThenticate (FUTA library). Local check vs the author's own asthma and brain-tumour papers: 0.20% and 0.00%, the remainder being the affiliation block only.
- **Page layout not visually verified** — LibreOffice is not installed on this machine, so page breaks around the 4 tables and 4 figures were never inspected. Open in Word and scroll before uploading.
- **Author block** currently matches the asthma paper (name, department, corresponding email). Change if a different form is wanted.

## Compliance check (verified programmatically)
- Title 13 words, no acronyms (limit 15).
- Abstract 242 words, no citations (limit 250).
- Structure: Introduction – Method – Results and Discussion – Conclusion.
- Contribution stated in §1.1; research questions in §1.2; related-work table as Table 1.
- 37 references, all cited in text, IEEE style with DOIs; 32 published 2021 or later (journal asks ≥30 from the last 5 years).
- ~5,100 words plus 4 tables and 4 figures (journal minimum 8 pages).

## Citation verification status
- All 37 reference entries were generated from Crossref metadata, not typed by hand.
- [9] Alzubaidi et al.: metadata verified; **full text not read** (publisher returned 403). Their ">98%" is cited as reported by two independent secondary sources; their protocol is deliberately not described.
- [10] Jennifer et al.: augmentation to 10,002 images is stated at dataset level; the order relative to train/test splitting is **not reported** — the manuscript says exactly that and does not allege leakage.
- One candidate reference was excluded because it is **retracted**.

## Claims to keep calibrated
- The residual gap must not be attributed to curation alone — curated cells (IDB1) and smear cells (IDB2) come from different source images. Stated in §3.7.
- Absolute curated accuracy (91–93%) is below published values by design; this is a relative comparison under a fixed simple pipeline. Stated in §3.6.
- The residual network interval is a fold range over 10 draws, not a paired percentile interval.
