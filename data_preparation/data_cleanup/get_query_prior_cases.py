#!/usr/bin/env python

"""Get prior cases of queries from a given text file. Each line in the text
file provides information about a query-case pair, indicating whether the case
is a prior case for the given query (0 for NOT a prior case and 1 for prior
case). The format is as follows:

    qID Q0 cID 0/1
    |   |   |   |
    |   |   |   |-- 0 for NOT a prior case, 1 for prior case
    |   |   |------ Case ID
    |   |---------- Redundant Column
    |-------------- Query ID

Further details about the formatting can be found in the README.md file of the
data file that can be downloaded from the link below:
https://zenodo.org/record/4063986/files/AILA_2019_dataset.zip?download=1

Due to the simplicity of the task, no commandline interface is created.
"""

import json
from collections import defaultdict

# Path to file with prior cases
path = ("/home/workboots/Datasets/AILA_2019/prior_case_retrieval/"
        "relevance_judgments_priorcases.txt")

# Path to save query prior cases
savepath = "query_prior_cases.json"

prior_cases = defaultdict(lambda: list())

with open(path, 'r') as f:
    for pline in f:
        qid, _, cid, isprior = pline.split()
        if int(isprior) == 0:
            continue
        prior_cases[qid].append(cid)

with open(savepath, 'w') as f:
    json.dump(prior_cases, f, indent=4)
