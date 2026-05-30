import argparse
from pathlib import Path
from seqeval.metrics import classification_report


import pandas as pd
import spacy


# spaCy uses different entity labels than the dataset
# This maps spaCy labels to the simpler course labels
SPACY_TO_COURSE_LABEL = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "NORP": "MISC",
    "LANGUAGE": "MISC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
}


def read_ner_file(path):
    """
    Reads the NER test file

    Expected columns:
    sentence id, token id, token, BIO NER tag
    """
    df = pd.read_csv(path, sep="\t")

    expected_columns = {"sentence id", "token id", "token", "BIO NER tag"}
    missing = expected_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df


def make_sentences(df):
    """
    Groups the tokens back into sentences

    Returns a list of dictionaries:
    {
        "sentence_id": ...,
        "tokens": [...],
        "gold_tags": [...]
    }
    """
    sentences = []

    for sentence_id, group in df.groupby("sentence id", sort=False):
        group = group.sort_values("token id")

        tokens = group["token"].astype(str).tolist()
        gold_tags = group["BIO NER tag"].astype(str).tolist()

        sentences.append({
            "sentence_id": sentence_id,
            "tokens": tokens,
            "gold_tags": gold_tags,
        })

    return sentences


def predict_bio_tags(nlp, tokens):
    """
    Runs spaCy NER on a tokenized sentence and converts the output to BIO tags

    We use spaCy's Doc object so that spaCy keeps the original tokenization
    from the dataset
    """
    spaces = [True] * len(tokens)
    if spaces:
        spaces[-1] = False

    doc = spacy.tokens.Doc(nlp.vocab, words=tokens, spaces=spaces)
    doc = nlp(doc)

    predicted_tags = ["O"] * len(tokens)

    for ent in doc.ents:
        label = SPACY_TO_COURSE_LABEL.get(ent.label_)

        # Ignore labels that are not part of the course NER scheme,
        # such as DATE, TIME, MONEY, CARDINAL, etc.
        if label is None:
            continue

        for i, token in enumerate(ent):
            if i == 0:
                predicted_tags[token.i] = f"B-{label}"
            else:
                predicted_tags[token.i] = f"I-{label}"

    return predicted_tags


def evaluate_token_level(gold_tags, predicted_tags):
    """
    Simple token-level evaluation

    This is not a perfect entity-level NER evaluation
    but it is useful for a first quantitative analysis
    """
    correct = 0
    total = 0

    label_counts = {}

    for gold, pred in zip(gold_tags, predicted_tags):
        total += 1

        if gold == pred:
            correct += 1

        if gold not in label_counts:
            label_counts[gold] = {"total": 0, "correct": 0}

        label_counts[gold]["total"] += 1

        if gold == pred:
            label_counts[gold]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    return accuracy, label_counts


def save_predictions(sentences, output_path):
    """
    Saves token-level predictions to a TSV file
    """
    rows = []

    for sentence in sentences:
        sentence_id = sentence["sentence_id"]
        tokens = sentence["tokens"]
        gold_tags = sentence["gold_tags"]
        predicted_tags = sentence["predicted_tags"]

        for token_id, (token, gold, pred) in enumerate(
            zip(tokens, gold_tags, predicted_tags),
            start=1
        ):
            rows.append({
                "sentence id": sentence_id,
                "token id": token_id,
                "token": token,
                "gold tag": gold,
                "predicted tag": pred,
                "correct": gold == pred,
            })

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def save_errors(sentences, output_path):
    """
    Saves only the incorrect predictions.

    This file is useful for the qualitative error analysis in the poster.
    """
    rows = []

    for sentence in sentences:
        sentence_id = sentence["sentence_id"]
        tokens = sentence["tokens"]
        gold_tags = sentence["gold_tags"]
        predicted_tags = sentence["predicted_tags"]

        sentence_text = " ".join(tokens)

        for token_id, (token, gold, pred) in enumerate(
            zip(tokens, gold_tags, predicted_tags),
            start=1
        ):
            if gold != pred:
                rows.append({
                    "sentence id": sentence_id,
                    "token id": token_id,
                    "sentence": sentence_text,
                    "token": token,
                    "gold tag": gold,
                    "predicted tag": pred,
                })

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def print_label_results(label_counts):
    """
    Prints accuracy for each gold label.
    """
    print("\nAccuracy by gold label:")

    for label, counts in sorted(label_counts.items()):
        total = counts["total"]
        correct = counts["correct"]
        accuracy = correct / total if total > 0 else 0

        print(f"{label:10s} {accuracy:.3f} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Run spaCy NERC on a BIO-tagged TSV file.")
    parser.add_argument(
        "--input",
        default="NER-test.tsv",
        help="Path to the NER test TSV file."
    )
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model to use. Default: en_core_web_sm"
    )
    parser.add_argument(
        "--predictions-output",
        default="ner_predictions.tsv",
        help="Where to save all predictions."
    )
    parser.add_argument(
        "--errors-output",
        default="ner_errors.tsv",
        help="Where to save incorrect predictions."
    )

    args = parser.parse_args()

    print("Loading data...")
    df = read_ner_file(args.input)
    sentences = make_sentences(df)

    print(f"Loaded {len(sentences)} sentences.")
    print(f"Loading spaCy model: {args.model}")

    nlp = spacy.load(args.model)

    all_gold_tags = []
    all_predicted_tags = []

    print("Running NERC...")

    for sentence in sentences:
        predicted_tags = predict_bio_tags(nlp, sentence["tokens"])
        sentence["predicted_tags"] = predicted_tags

        all_gold_tags.extend(sentence["gold_tags"])
        all_predicted_tags.extend(predicted_tags)

    accuracy, label_counts = evaluate_token_level(all_gold_tags, all_predicted_tags)

    print("\nOverall token-level accuracy:")
    print(f"{accuracy:.3f}")

    print_label_results(label_counts)

    print("\nEntity-level evaluation (seqeval):")
    print(classification_report([all_gold_tags], [all_predicted_tags]))

    save_predictions(sentences, args.predictions_output)
    save_errors(sentences, args.errors_output)

    print("\nDone.")
    print(f"Saved predictions to: {args.predictions_output}")
    print(f"Saved errors to: {args.errors_output}")


if __name__ == "__main__":
    main()