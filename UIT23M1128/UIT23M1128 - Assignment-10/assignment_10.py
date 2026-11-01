# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
# Assignment 10: Prompt Engineering Techniques for LLMs

"""
Assignment 10: Prompt Engineering Techniques for LLMs
- Programmatic implementation of Zero-Shot, One-Shot, Few-Shot, and Chain-of-Thought (CoT) prompts
- Legal document summarization, risk clause detection, and plain-language brief generation
- Systematic evaluation of prompt templates and output quality impact
"""

import os
import json
from typing import Dict

def build_zero_shot_prompt(contract_text: str) -> str:
    """Builds a zero-shot prompt for contract analysis."""
    return f"""System: You are an expert legal assistant. Summarize the contract and identify risk clauses.
Contract Text:
"{contract_text}"
Summary & Risk Clauses:"""

def build_few_shot_prompt(contract_text: str) -> str:
    """Builds a few-shot prompt with exemplar input-output pairs."""
    return f"""System: Analyze legal contracts, extract key dates, liabilities, and plain-language summaries.

Example 1:
Contract: "The Licensee shall pay $50,000 annually. Failure to pay within 30 days results in immediate termination."
Output:
- Obligation: $50,000 annual fee.
- Risk Clause: 30-day default termination.

Example 2:
Contract: "{contract_text}"
Output:"""

def build_chain_of_thought_prompt(contract_text: str) -> str:
    """Builds a Chain-of-Thought (CoT) prompt enforcing step-by-step reasoning."""
    return f"""System: Perform legal analysis using explicit step-by-step reasoning.
Contract Text:
"{contract_text}"

Reasoning Steps:
1. Identify contracting parties and primary obligations.
2. Locate duration and confidentiality term limits.
3. Extract financial risk caps or liability limitations.

Final Analysis & Brief:"""

def evaluate_prompt_strategies(contract_text: str) -> Dict[str, str]:
    """Generates prompt templates across zero-shot, few-shot, and CoT strategies."""
    return {
        "Zero-Shot": build_zero_shot_prompt(contract_text),
        "Few-Shot": build_few_shot_prompt(contract_text),
        "Chain-of-Thought (CoT)": build_chain_of_thought_prompt(contract_text)
    }

def main():
    print("=" * 60)
    print("ASSIGNMENT 10: PROMPT ENGINEERING TECHNIQUES FOR LLMS")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__),"sample_contracts.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    sample_contract = contracts[0]["text"]
    print(f"\nSample Legal Contract:\n\"{sample_contract}\"\n")

    prompts = evaluate_prompt_strategies(sample_contract)

    print("=" * 60)
    for strategy_name, prompt_str in prompts.items():
        print(f"--- Prompting Strategy: {strategy_name} ---")
        print(prompt_str)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
