#!/usr/bin/env python

"""Script to separate texts of queries. Queries are provided in a single file
with each line corresponding to one query. The format of each line provides the
Query ID and its text in the following format:

    qID_1 || Text
    qID_2 || Text

Further details about the formatting can be found in the README.md file of the
data file that can be downloaded from the link below:
https://zenodo.org/record/4063986/files/AILA_2019_dataset.zip?download=1

Due to the simplicity of the task, no commandline interface is created.
"""

import os

# Path to load query text file
path = "/home/workboots/Datasets/AILA_2019/query_docs.txt"

# Relative path from current working directory to save texts
savepath = "query_texts/"

queries = {}

with open(path, 'r') as f:
    for qline in f:
        qid, qtext = qline.split("||")
        queries[qid.strip()] = qtext.strip()

if not os.path.exists(os.path.join(os.getcwd(), savepath)):
    os.makedirs(os.path.join(os.getcwd(), savepath))

for qid, qtext in queries.items():
    with open(os.path.join(os.getcwd(), savepath, f"{qid}.txt"), 'w') as f:
        f.write(qtext)
