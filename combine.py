# this code is used to combine undreadctor.tsv and my 90 redactions saved under output12.tsv

import glob

read_files = glob.glob("*.tsv")

with open("final.tsv", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
