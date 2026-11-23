"""ClassificationAgent: soft-voting ensemble of Random Forest, XGBoost and SVM
(case study section 6.2), with a confidence gate and SHAP explanations."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

from .data import CLASSES, make_vignettes
from .ner import FEATURES, LABELS, extract_entities

CONFIDENCE_GATE = 0.72   # below this the answer is 'Indeterminate' (case study 6.2, 7.3)


class EnsembleClassifier:
    def __init__(self, seed=7):
        self.seed = seed
        self.model = None
        self.explainer = None
        self.metrics = {}

    @staticmethod
    def featurise(notes):
        return pd.DataFrame([extract_entities(n) for n in notes], columns=FEATURES)

    def fit(self, n=1500):
        df = make_vignettes(n=n, seed=self.seed)
        X, y = self.featurise(df.note), df.label.map({c: i for i, c in enumerate(CLASSES)}).values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.seed)
        rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=self.seed)
        xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, eval_metric="mlogloss",
                            random_state=self.seed, verbosity=0)
        svm = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=self.seed), cv=3, ensemble=False))
        self.model = VotingClassifier([("rf", rf), ("xgb", xgb), ("svm", svm)], voting="soft").fit(Xtr, ytr)
        proba = self.model.predict_proba(Xte)
        pred = proba.argmax(1)
        gated = proba.max(1) >= CONFIDENCE_GATE
        self.metrics = {
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "per_class_f1": dict(zip(CLASSES, map(float, f1_score(yte, pred, average=None)))),
            "auroc_ovr": float(roc_auc_score(yte, proba, multi_class="ovr")),
            "coverage_after_gate": float(gated.mean()),
            "accuracy_when_confident": float(accuracy_score(yte[gated], pred[gated])) if gated.any() else float("nan"),
            "confusion": confusion_matrix(yte, pred).tolist(),
        }
        self._test = (Xte, yte, proba)
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model.named_estimators_["xgb"])
        except Exception:
            self.explainer = None
        return self

    def predict(self, entities: dict) -> dict:
        x = pd.DataFrame([entities], columns=FEATURES)
        proba = self.model.predict_proba(x)[0]
        k = int(proba.argmax())
        conf = float(proba[k])
        label = CLASSES[k] if conf >= CONFIDENCE_GATE else "Indeterminate"
        return {"prediction": label, "top_class": CLASSES[k], "confidence": conf,
                "probabilities": {c: float(p) for c, p in zip(CLASSES, proba)},
                "feature_importances": self.explain(x, k)}

    def explain(self, x, k, top=5) -> dict:
        """SHAP contributions (from the XGBoost member) for the top class, largest first."""
        if self.explainer is None:
            return {}
        sv = self.explainer.shap_values(x)
        vals = np.asarray(sv)
        if vals.ndim == 3:          # (rows, features, classes)
            vals = vals[0, :, k]
        elif isinstance(sv, list):
            vals = np.asarray(sv[k])[0]
        else:
            vals = vals[0]
        order = np.argsort(-np.abs(vals))[:top]
        return {LABELS[FEATURES[i]]: round(float(vals[i]), 3) for i in order}
