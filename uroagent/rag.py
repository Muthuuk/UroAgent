"""GuidelineAgent: retrieval over clinical guideline knowledge cards (case study 5.3).

Runs on a laptop or free cloud CPU with no API keys. Six stages, mirroring the case study:
1 query understanding (entities) -> 2 multi-query generation -> 3 hybrid retrieval
(BM25 keyword + TF-IDF vector similarity, fused by Reciprocal Rank Fusion) ->
4 de-duplication -> 5 re-ranking (condition match) -> 6 contextual compression
(keep the most relevant sentence of each card).
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "guideline_cards.json"
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in for with on at is are be by as if it this that from use who when than any".split())


def tok(text):
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        q = tok(query)
        out = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            out.append(s)
        return np.array(out)


SYNONYMS = {
    "OAB": "overactive bladder urgency frequency nocturia urge incontinence bladder training",
    "BPS": "bladder pain syndrome interstitial cystitis pain filling hunner",
    "UTI": "urinary tract infection burning dysuria cloudy urine culture antibiotic",
}


INTENTS = {
    "treatment": ["treat", "manag", "first-line", "first line", "therapy", "what should", "what do i do", "next step", "plan"],
    "assessment": ["diagnos", "criteria", "assess", "definition", "diary", "dipstick", "test", "investigat", "confirm"],
    "medicine": ["medicine", "drug", "side effect", "risks", "antibiotic", "mirabegron", "antimuscarinic", "prescri", "medication", "tablet"],
    "safety": ["refer", "red flag", "blood", "haematuria", "worse", "urgent", "safety"],
}


def query_intents(query):
    q = query.lower()
    return {k for k, words in INTENTS.items() if any(w in q for w in words)}


class GuidelineRetriever:
    def __init__(self, path=KB_PATH):
        data = json.loads(Path(path).read_text())
        self.cards = data["cards"]
        self.note = data["_note"]
        corpus = [f"{c['section']} {c['text']} {' '.join(c['conditions'])}" for c in self.cards]
        self.bm25 = BM25(corpus)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.tfidf.fit_transform(corpus)

    def _multi_query(self, query, condition=None, findings=()):
        qs = [query]
        if condition in SYNONYMS:
            qs.append(f"{query} {SYNONYMS[condition]}")
            qs.append(f"{condition} management first-line treatment assessment")
        if findings:
            qs.append(" ".join(findings))
        return qs

    def search(self, query, condition=None, findings=(), k=5):
        rrf = Counter()
        for q in self._multi_query(query, condition, findings):
            dense = (self.matrix @ self.tfidf.transform([q]).T).toarray().ravel()
            sparse = self.bm25.scores(q)
            for ranking in (np.argsort(-dense)[:8], np.argsort(-sparse)[:8]):
                for rank, i in enumerate(ranking):
                    rrf[int(i)] += 1.0 / (60 + rank)          # Reciprocal Rank Fusion
        candidates = [i for i, _ in rrf.most_common(16)]       # de-duplicated by card id
        intents = query_intents(query)
        def rerank(i):                                         # stage 5: condition and intent match
            card = self.cards[i]
            boost = 0.03 if condition and condition in card["conditions"] else 0.0
            boost += 0.045 * len(intents & set(card.get("intent", [])))
            return rrf[i] + boost
        top = sorted(candidates, key=rerank, reverse=True)[:k]
        return [self._compress(self.cards[i], query, rerank(i)) for i in top]

    def _compress(self, card, query, score):
        sentences = re.split(r"(?<=[.;])\s+", card["text"])
        q = set(tok(query))
        best = max(sentences, key=lambda s: len(q & set(tok(s))))
        return {**card, "score": round(float(score), 4), "snippet": best,
                "citation": f"{card['source']} ({card['version']}), {card['section']}"}
