# DOCX tables check (20261013_103623)

Before (`extract.py` and `hidden_text.py` at HEAD, d8f61af: DOCX tables not read) → after (tables read in document order, one row per line, cells separated by tabs). Local, no API. Made by `eval/check_docx_tables.py --compare-with HEAD --migraine <migraine PDF>`; the DOCX files are rebuilt from the papers' PDFs with their tables as Word tables.

### vitamin D (1 table(s), 1724 characters in them)

| Measure | Before | After |
|---|---|---|
| Characters | 22382 | 24359 |
| Tokens | 5729 | 6593 |
| Mode | full | full |
| RAG chunks | 26 | 28 |
| Full-context sections | 10 | 11 |
| Table-row quotes found | 0/23 | 23/23 |
| Table rows cut by a chunk | 0/0 | 0/23 |

Sections now: Front matter, Abstract, Introduction, Methods, Results, Discussion, Limitations, Conclusion, Declarations, Appendix, References

### migraine (2 table(s), 2379 characters in them)

| Measure | Before | After |
|---|---|---|
| Characters | 38273 | 40885 |
| Tokens | 9769 | 11120 |
| Mode | full | full |
| RAG chunks | 47 | 51 |
| Full-context sections | 12 | 12 |
| Table-row quotes found | 0/40 | 40/40 |
| Table rows cut by a chunk | 0/0 | 0/53 |

Sections now: Front matter, Introduction, Methods, Results, Discussion, Conclusion, Declarations, References

