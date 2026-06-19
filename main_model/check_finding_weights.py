"""
Verification: check that prepare_labels_and_weights correctly marks tokens
inside positive-finding sentences.  Loads only the tokenizer — no model needed.

Run: python check_finding_weights.py
"""
import re
import torch
from transformers import AutoTokenizer

from data_utils_compat import extract_disease_labels

TOKENIZER_NAME = "razent/SciFive-base-Pubmed_PMC"
FINDING_W = 4.0

# ---- Sample abnormal reports (hand-picked patterns) ----
SAMPLES = [
    # 1. Single finding
    "the heart size is mildly enlarged suggesting cardiomegaly. the lungs are clear without focal consolidation. no pleural effusion.",
    # 2. Multiple findings
    "there is cardiomegaly and mild pulmonary edema. the costophrenic angles are blunted consistent with small bilateral pleural effusions.",
    # 3. Finding inside a complex sentence
    "patchy opacities in the right lower lobe may represent consolidation. the cardiac silhouette is normal in size.",
    # 4. ALL normal (should have NO weighted tokens)
    "the lungs are clear without focal consolidation. the heart size is normal. no pleural effusion or pneumothorax is identified.",
    # 5. Finding sentence with negation that should NOT be weighted
    "no evidence of cardiomegaly or pulmonary edema. the lungs are clear. small left pleural effusion is noted.",
]


def check(reports, tokenizer, finding_w=FINDING_W):
    enc = tokenizer(
        reports, return_tensors="pt", padding=True, truncation=True,
        max_length=200, return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets   = enc["offset_mapping"]      # [B, T, 2]
    attn      = enc["attention_mask"]      # [B, T]

    B, T = input_ids.shape
    weights = torch.ones(B, T, dtype=torch.float)

    for b, report in enumerate(reports):
        # ---- Find positive-finding sentence spans ----
        spans, pos = [], 0
        for sent in re.split(r'(?<=[.!?])\s+', report):
            if not sent:
                continue
            start = report.find(sent, pos)
            if start < 0:
                start = pos
            end = start + len(sent)
            pos = end
            is_pos = extract_disease_labels(sent.lower())
            if is_pos:
                spans.append((start, end))
            # Print sentence-level diagnosis
            tag = f"  [+] {is_pos}" if is_pos else "  [-] (neg/normal)"
            print(f"  SENT [{start:3d}:{end:3d}]{tag}  {sent[:90]}")

        if not spans:
            print(f"  => NO positive-finding sentences. All weights=1.0 (correct)\n")
            continue

        print(f"  => {len(spans)} positive-finding sentence(s), spans={spans}")

        # ---- Apply weights ----
        for t in range(T):
            if attn[b, t] == 0:
                continue
            cs, ce = int(offsets[b, t, 0]), int(offsets[b, t, 1])
            if cs == ce:
                continue
            if any(cs >= s and ce <= e for (s, e) in spans):
                weights[b, t] = finding_w

        # ---- Decode weighted tokens ----
        print(f"  Weighted tokens (weight={finding_w}):")
        found_any = False
        for t in range(T):
            w = weights[b, t].item()
            if w > 1.0:
                found_any = True
                tok = tokenizer.decode(input_ids[b, t].item())
                cs, ce = int(offsets[b, t, 0]), int(offsets[b, t, 1])
                print(f"    token[{t:3d}]  chars[{cs:3d}:{ce:3d}]  {tok!r}")
        if not found_any:
            print(f"    ⚠️  NONE — offset_mapping may be broken (all (0,0)?)")
        print()


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Fast: {tokenizer.is_fast}")
    print()

    for i, report in enumerate(SAMPLES):
        print(f"{'='*70}")
        print(f"[Sample {i+1}] {report[:100]}...")
        print(f"{'='*70}")
        check([report], tokenizer)

    # Also test batched (multiple reports together)
    print(f"{'='*70}")
    print(f"[Batched] All 5 samples together")
    print(f"{'='*70}")
    check(SAMPLES, tokenizer)


if __name__ == "__main__":
    main()
