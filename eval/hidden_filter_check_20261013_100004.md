# Hidden-text filter check (20261013_100004)

Before (`hidden_text.py` at HEAD, 4b720ed) → after (this change). Local, no API. Made by `eval/check_hidden_filter.py --compare-with HEAD`.
Instruction left: runs of 3 consecutive words of the instruction still in the filtered text. Removed chars: raw minus filtered length, with the instruction's length in brackets.

## Files made by eval/make_injection_samples.py

| File | Marker left | Instruction left (3-word runs) | Removed chars (instruction) | Pieces |
|---|---|---|---|---|
| hidden_ar_blunt.docx | no | 0/7 | 60 (60) | 1 |
| hidden_ar_polite.docx | no | 0/14 | 101 (101) | 1 |
| hidden_en_blunt.docx | no | 0/10 | 81 (81) | 1 |
| hidden_en_polite.docx | no | 0/19 | 122 (122) | 1 |
| white_ar_blunt.pdf | no | 4/7 → 0/7 | 10 → 58 (60) | 1 → 9 |
| white_ar_polite.pdf | no | 11/14 → 0/14 | 10 → 99 (101) | 1 → 16 |
| white_en_blunt.pdf | no | 0/10 | 78 → 80 (81) | 2 |
| white_en_polite.pdf | no | 0/19 | 117 → 121 (122) | 3 |

## Hand-made files (Word, 2026-10-04; kept locally) and clean papers

| File | Marker left | Instruction left (3-word runs) | Removed chars (instruction) | Pieces |
|---|---|---|---|---|
| hidden_ar_blunt.docx | no | 0/7 | 9658 (60) | 2796 |
| hidden_ar_polite.docx | no | 0/14 | 9692 (101) | 2795 |
| hidden_en_blunt.docx | no | 0/10 | 9673 (81) | 2793 |
| hidden_en_polite.docx | no | 0/19 | 9714 (122) | 2793 |
| white_ar_blunt.pdf | no | 4/7 → 0/7 | 5 → 53 (60) | 1 → 3 |
| white_ar_polite.pdf | no | 10/14 → 0/14 | 5 → 94 (101) | 1 → 3 |
| white_en_blunt.pdf | no | 0/10 | 76 → 81 (81) | 3 |
| white_en_polite.pdf | no | 0/19 | 115 → 123 (122) | 4 |
| 1-s2.0-S0021755720302072-main.pdf (clean) | — | — | 0 | 0 |
| Association_of_Migraine_and_Irritable_Bowel_Syndro.pdf (clean) | — | — | 0 | 0 |
