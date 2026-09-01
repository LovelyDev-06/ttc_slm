"""Reasoning verifier: scores coherence/answer confidence; final correctness is exact option match."""
import re, numpy as np
from src.reasoning_utils import extract_final_answer, evaluate_output

def heuristic_score(reasoning, problem):
    if not reasoning.strip(): return 0.0
    score=0.0
    if extract_final_answer(reasoning,problem.get("choices")): score+=0.45
    if len(reasoning.split())>=12: score+=0.2
    if re.search(r"because|therefore|thus|since|so",reasoning,re.I): score+=0.2
    if len(reasoning)<4000: score+=0.15
    return min(score,1.0)

def verify_final(reasoning,problem):
    ev=evaluate_output(reasoning,problem); ev["quality_score"]=heuristic_score(reasoning,problem); return ev
