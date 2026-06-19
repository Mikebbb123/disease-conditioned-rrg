"""
Disease label extraction — regex-based.
P3: 5-class binary labels (cardiomegaly, pulmonary edema, consolidation,
atelectasis, pleural effusion).
"""
import re
from typing import List
import torch

DISEASE_PATTERNS = {
    "cardiomegaly": {
        "positive": [
            r"\bcardiomegaly\b",
            r"\b(?:heart|cardiac silhouette)\s+(?:is\s+)?(?:mildly\s+|moderately\s+|markedly\s+|borderline\s+)?(?:enlarged|increased in size)",
            r"\benlarged heart\b",
            r"\bcardiac enlargement\b",
            r"\bincreased cardiac (?:silhouette|size)\b",
        ],
        "negative": [
            r"\bno\s+(?:[\w,;]+\s+){0,5}cardiomegaly\b",
            r"\bwithout\s+(?:[\w,;]+\s+){0,5}cardiomegaly\b",
            r"\bfree of\s+(?:[\w,;]+\s+){0,5}cardiomegaly\b",
            r"\bclear of\s+(?:[\w,;]+\s+){0,5}cardiomegaly\b",
            r"\b(?:heart|cardiac silhouette)(?:\s+size)?\s+(?:is\s+)?normal\b",
            r"\bnormal\s+(?:heart|cardiac)(?:\s+silhouette)?\b",
            r"\bnegative for\s+(?:[\w,;]+\s+){0,4}cardiomegaly\b",
        ],
    },
    "pulmonary edema": {
        "positive": [
            r"\b(?:pulmonary\s+)?edema\b",
            r"\bvascular congestion\b",
            r"\bcongestion\b",
            r"\bcephalization\b",
        ],
        "negative": [
            r"\bno\s+(?:[\w,;]+\s+){0,5}(?:edema|congestion|cephalization)\b",
            r"\bwithout\s+(?:[\w,;]+\s+){0,5}(?:edema|congestion)\b",
            r"\b(?:or|nor)\s+(?:[\w,;]+\s+){0,4}(?:edema|congestion)\b",
            r"\bnegative for\s+(?:[\w,;]+\s+){0,4}(?:edema|congestion)\b",
            r"\bfree of\s+(?:[\w,;]+\s+){0,5}(?:edema|congestion|cephalization)\b",
            r"\bclear of\s+(?:[\w,;]+\s+){0,5}(?:edema|congestion|cephalization)\b",
        ],
    },
    "consolidation": {
        "positive": [
            r"\bconsolidation[s]?\b",
            r"\bconsolidative\b",
        ],
        "negative": [
            r"\bno\s+(?:[\w,;]+\s+){0,5}(?:consolidation[s]?|consolidative)\b",
            r"\bwithout\s+(?:[\w,;]+\s+){0,5}(?:consolidation[s]?|consolidative)\b",
            r"\b(?:or|nor)\s+(?:[\w,;]+\s+){0,4}(?:consolidation[s]?|consolidative)\b",
            r"\bnegative for\s+(?:[\w,;]+\s+){0,4}(?:consolidation[s]?|consolidative)\b",
            r"\bfree of\s+(?:[\w,;]+\s+){0,4}(?:consolidation[s]?|consolidative)\b",
            r"\bclear of\s+(?:[\w,;]+\s+){0,5}(?:consolidation[s]?|consolidative)\b",
        ],
    },
    "atelectasis": {
        "positive": [r"\batelectasis\b", r"\batelectatic\b"],
        "negative": [
            r"\bno\s+(?:[\w,;]+\s+){0,5}atelectasis\b",
            r"\bwithout\s+(?:[\w,;]+\s+){0,5}atelectasis\b",
            r"\bnegative for\s+(?:[\w,;]+\s+){0,4}atelectasis\b",
            r"\bfree of\s+(?:[\w,;]+\s+){0,5}atelectasis\b",
            r"\bclear of\s+(?:[\w,;]+\s+){0,5}atelectasis\b",
        ],
    },
    "pleural effusion": {
        "positive": [
            r"\bpleural effusion[s]?\b",
            r"\beffusion[s]?\b",
            r"\bcostophrenic blunting\b",
            r"\bblunting of the costophrenic\b",
        ],
        "negative": [
            r"\bno\s+(?:[\w,;]+\s+){0,5}effusion[s]?\b",
            r"\bwithout\s+(?:[\w,;]+\s+){0,5}effusion[s]?\b",
            r"\b(?:or|nor)\s+(?:[\w,;]+\s+){0,4}effusion[s]?\b",
            r"\bnegative for\s+(?:[\w,;]+\s+){0,4}effusion[s]?\b",
            r"\bfree of\s+(?:[\w,;]+\s+){0,4}effusion[s]?\b",
            r"\bclear of\s+(?:[\w,;]+\s+){0,5}effusion[s]?\b",
            r"\bno\s+(?:[\w,;]+\s+){0,5}costophrenic blunting\b",
        ],
    },
}

_COMPILED = {
    d: {
        "positive": [re.compile(p) for p in pat["positive"]],
        "negative": [re.compile(p) for p in pat["negative"]],
    }
    for d, pat in DISEASE_PATTERNS.items()
}
DISEASE_LIST = list(DISEASE_PATTERNS.keys())


def _find_spans(text, compiled_patterns):
    spans = []
    for cp in compiled_patterns:
        for m in cp.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _is_covered(pos_span, neg_spans, margin=5):
    ps, pe = pos_span
    for ns, ne in neg_spans:
        if ns - margin <= ps and pe <= ne + margin:
            return True
    return False


def extract_disease_labels(report: str, patterns=None):
    if patterns is None:
        compiled = _COMPILED
        diseases = DISEASE_LIST
    else:
        compiled = {
            d: {
                "positive": [re.compile(p) for p in pat["positive"]],
                "negative": [re.compile(p) for p in pat["negative"]],
            }
            for d, pat in patterns.items()
        }
        diseases = list(patterns.keys())

    text = report.lower()
    findings = []
    for disease in diseases:
        pat = compiled[disease]
        pos_spans = _find_spans(text, pat["positive"])
        if not pos_spans:
            continue
        neg_spans = _find_spans(text, pat["negative"])
        if not neg_spans:
            findings.append(disease)
            continue
        if any(not _is_covered(ps, neg_spans) for ps in pos_spans):
            findings.append(disease)
    return findings


def report_to_multihot(report: str) -> torch.Tensor:
    findings = set(extract_disease_labels(report))
    return torch.tensor([1.0 if d in findings else 0.0 for d in DISEASE_LIST], dtype=torch.float)


