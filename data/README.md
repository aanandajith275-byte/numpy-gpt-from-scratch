# Training text

`abstracts.txt` holds the 1,398 non-empty abstracts of the Cranfield collection (aerodynamics and
aeronautical-engineering papers, mostly 1950s–60s), one abstract per line, whitespace normalised:
1.43 million characters in total. The text is lower-case in the source; the loader lower-cases the
handful of stray capitals.

The Cranfield collection was assembled by Cyril Cleverdon and colleagues at the College of
Aeronautics, Cranfield, and has been freely redistributed for research and teaching for decades.
This copy was extracted from the TREC-formatted version at
[oussbenk/cranfield-trec-dataset](https://github.com/oussbenk/cranfield-trec-dataset).

The split is by **document**: 10% of the abstracts (seed 0) are held out for validation, so the
model is always scored on abstracts it has never seen.
