import csv
import itertools

import nltk
from nltk.corpus import wordnet as wn


def ycb_google_16k_meta_filename_to_word(filename: str):
    words = filename.replace(".glb", "")[4:].replace("_", " ").split()

    if len(words) == 1:
        return words[0], " ".join(words)

    if {"box", "lego", "toy", "pitcher"}.intersection(words):
        return words[-2], " ".join(words)

    return words[-1], " ".join(words)


ycb_synsets = {
    "container.n.01",
    "bread.n.01",
    "food",
    "fruit.n.01",
    "cooking_utensil.n.01",
    "tableware.n.01",
    "cutlery.n.02",
    "tool.n.01",
    "lever.n.01'",
    "lock.n.01",
    "hand_tool.n.01",
    "device.n.01",
    "equipment.n.01",
    "cube.n.05",
    "game.n.01",
    "artifact.n.01",
}


def ycb_google_16k_meta_get_synset(word: str, object_name: str):
    gen_synsets = {synset for synset in wn.synsets(word)}
    gen_synsets = gen_synsets.union(
        {hyper for synset in gen_synsets for hyper in synset.hypernyms()}
    )
    gen_synsets = gen_synsets.union(
        {hyper for synset in gen_synsets for hyper in synset.hypernyms()}
    )
    gen_synsets = {synset.name() for synset in gen_synsets}
    # two_level_synsets = {synset.name() for synset in wn.synsets(word)}.union(
    #     {hyper.name() for synset in wn.synsets(word) for hyper in synset.hypernyms()})
    key_synsets = ycb_synsets.intersection(gen_synsets)
    if key_synsets:
        return key_synsets.pop()
    return None


with open(
    "habitat/datasets/object_nav/create/ycb_google_16k_meta.csv", "r"
) as f:
    csv_r = csv.reader(f)
    headers = next(csv_r)
    print(headers)
    key_ind = headers.index("filename")
    rows = [
        (line[key_ind], ycb_google_16k_meta_filename_to_word(line[key_ind]))
        for line in csv_r
        if line[key_ind] != ""
    ]
    rows_synset = [
        (filename, word[1], word[0], wn.synsets(word[0]))
        for filename, word in rows
    ]
    synsets_source = [
        (
            filename,
            word[1],
            word[0],
            ycb_google_16k_meta_get_synset(word[0], word[1]),
            # wn.synsets(word[0])[0].definition(), wn.synsets(word[0])[0].hypernyms(),
            # wn.synsets(word[0])[0].hypernyms()[0].hypernyms()
        )
        for filename, word in rows
    ]
    # [(line[0], wn.synset(line[key_ind])) for line in csv_r if line[key_ind] != '']
    for row in synsets_source:
        print(f"{row}")

    with open(
        "habitat/datasets/object_nav/create/ycb_google_16k_meta_output.csv",
        "w",
    ) as of:
        writer = csv.writer(of)
        writer.writerow(["filename"])
        for row in synsets_source:
            writer.writerow(row)  # [row[0], row[1], row[0]])

# container.n.01
# bread.n.01
# food
# fruit.n.01
# cooking_utensil.n.01
# tableware.n.01
# cutlery.n.02
# tool.n.01
# lever.n.01'
# lock.n.01
# hand_tool.n.01
# device.n.01
# equipment.n.01
# cube.n.05
# game.n.01
# artifact.n.01

with open("category_mapping.tsv", "r") as f:
    csv_r = csv.reader(f, delimiter="\t")
    headers = next(csv_r)
    key_ind = headers.index("wnsynsetkey")
    print(headers)
    synsets_source = [
        (line[1], wn.synset(line[key_ind]))
        for line in csv_r
        if line[key_ind] != ""
    ]

with open("dictionaries.csv", "r") as f:
    csv_r = csv.reader(f)
    next(csv_r)
    next(csv_r)
    synsets_target = [
        [(line[1], sn) for sn in wn.synsets(line[1])] for line in csv_r
    ]
    synsets_target = list(itertools.chain(*synsets_target))

best_matches = []
for word1, sn1 in synsets_source:
    best = max(
        [
            [
                sn1.path_similarity(sn2, simulate_root=False),
                word1,
                word2,
                sn1,
                sn2,
            ]
            for word2, sn2 in synsets_target
            if sn1.path_similarity(sn2, simulate_root=False) is not None
        ]
    )
    best = best + [best[-2].lowest_common_hypernyms(best[-1])]
    best_matches.append(best)

print("Similarity,word1,word2,synset1,synset2,lcasynset")
for best_match in sorted(best_matches, key=lambda x: x[1]):
    # Only difference between the results posted is this if statement
    if best_match[-3] in best_match[-1] or best_match[-2] in best_match[-1]:
        print(best_match)
