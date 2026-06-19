"""RadGraph F1 evaluation for radiology report generation."""
from typing import List, Dict


def compute_radgraph_f1(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from radgraph import F1RadGraph
    except ImportError:
        raise ImportError("radgraph not installed. Run: pip install radgraph")

    f1radgraph = F1RadGraph(reward_level="all")
    mean_reward, _, _, _ = f1radgraph(hyps=predictions, refs=references)
    simple, partial, complete = mean_reward
    return {
        "RadGraph_Simple":   round(simple * 100, 2),
        "RadGraph_Partial":  round(partial * 100, 2),
        "RadGraph_Complete": round(complete * 100, 2),
    }
