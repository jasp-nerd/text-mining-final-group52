# Text Mining final project, Group 52

Final project for Text Mining for AI at VU Amsterdam (2026).

Three NLP tasks on the provided test sets: NERC, sentence-level sentiment, and sentence-level topic. Sentiment is the task where we compare more than one system for the deeper analysis. The full write-up is on the poster (handed in separately); this README is just a guide to the code.

## Group 52

- B.N.J. Bom (Boris)
- J.F.L. Meijerink (Jasper)
- K.C.E. Abdoellah (Kyan)
- S. Bakker (Skick)

## Running

Python 3.9, spaCy pinned to 3.7 like in the labs.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

Then open the notebooks in Jupyter and run them top to bottom. The BERT cells in `sentiment.ipynb` and `topic.ipynb` train on CPU and take a few minutes each.