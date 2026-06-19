"""
CheXbert-based clinical F1 evaluation using the f1chexbert pip package.

IMPORTANT: f1chexbert monkey-patches tokenizer._batch_encode_plus.
We save the original and restore it after each scoring call.
"""
from typing import List, Dict, Optional

import transformers as _transformers
_ORIGINAL_BEP = _transformers.PreTrainedTokenizerBase._batch_encode_plus


def _undo_tokenizer_patch():
    current = _transformers.PreTrainedTokenizerBase._batch_encode_plus
    if current is not _ORIGINAL_BEP:
        _transformers.PreTrainedTokenizerBase._batch_encode_plus = _ORIGINAL_BEP


_scorer = None


def get_chexbert_labeler(cache_dir: Optional[str] = None, device: str = "cuda"):
    global _scorer
    if _scorer is None:
        from f1chexbert import F1CheXbert
        print(f"[CheXbert] Loading F1CheXbert (device={device}) …")
        _scorer = F1CheXbert(device=device)
        _undo_tokenizer_patch()
        print("[CheXbert] Ready.")
    return _scorer


def compute_clinical_f1_chexbert(
    predictions: List[str],
    references: List[str],
    labeler=None,
) -> Dict[str, float]:
    if labeler is None:
        labeler = get_chexbert_labeler()

    accuracy, _, class_report_14, class_report_5 = labeler(
        hyps=predictions, refs=references,
    )
    _undo_tokenizer_patch()

    results = {
        "ClinicalF1_chexbert_accuracy":    round(accuracy * 100, 2),
        "ClinicalF1_chexbert_macro_F_14":  round(class_report_14["macro avg"]["f1-score"] * 100, 2),
        "ClinicalF1_chexbert_micro_F_14":  round(class_report_14["micro avg"]["f1-score"] * 100, 2)
                                            if "micro avg" in class_report_14 else None,
        "ClinicalF1_chexbert_macro_F_5":   round(class_report_5["macro avg"]["f1-score"] * 100, 2),
        "ClinicalF1_chexbert_micro_F_5":   round(class_report_5["micro avg"]["f1-score"] * 100, 2)
                                            if "micro avg" in class_report_5 else None,
        "ClinicalF1_chexbert_macro_F":     round(class_report_5["macro avg"]["f1-score"] * 100, 2),
    }

    five_classes = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    for cname in five_classes:
        if cname in class_report_5:
            results[f"CheXbert5_{cname}_F1"] = round(class_report_5[cname]["f1-score"] * 100, 2)

    return {k: v for k, v in results.items() if v is not None}
