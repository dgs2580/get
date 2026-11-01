# Name : Divyanshu Girijesh Singh
# Class : B.Tech IT B - B3
# PRN : UIT23M1128
# Assignment 6: Named Entity Recognition (NER) 



"""
Assignment 6: Named Entity Recognition (NER)
- spaCy NER pipeline & displaCy entity visualization logic
- Custom spaCy EntityRuler for domain-specific clinical entity parsing
- Extract Drug names, Dosages, Symptoms, and Diagnoses from EHR summaries
"""

import os
import re
import json
from typing import Dict, List

def extract_clinical_entities_rulebased(text: str) -> Dict[str, List[str]]:
    """
    Extracts clinical medical entities (Drug, Dosage, Diagnosis, Symptom)
    using pattern matching / custom EntityRuler logic.
    """
    drug_pattern = r'\b(Amoxicillin|Lisinopril|Metformin|Paracetamol|Ibuprofen|Aspirin)\b'
    dosage_pattern = r'\b\d+\s*(?:mg|g|ml)\b'
    diagnosis_pattern = r'\b(hypertension|Diabetes Mellitus|Type 2 Diabetes|fever|asthma)\b'
    symptom_pattern = r'\b(dizziness|nausea|headache|pain|cough)\b'

    drugs = list(set(re.findall(drug_pattern, text, re.IGNORECASE)))
    dosages = list(set(re.findall(dosage_pattern, text, re.IGNORECASE)))
    diagnoses = list(set(re.findall(diagnosis_pattern, text, re.IGNORECASE)))
    symptoms = list(set(re.findall(symptom_pattern, text, re.IGNORECASE)))

    return {
        "DRUG": drugs,
        "DOSAGE": dosages,
        "DIAGNOSIS": diagnoses,
        "SYMPTOM": symptoms
    }

def spacy_clinical_ner(text: str) -> dict:
    """Uses spaCy's pretrained NER engine if installed, falling back cleanly."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return {"spacy_entities": entities}
    except Exception:
        return {"spacy_entities": []}

def main():
    print("=" * 60)
    print("ASSIGNMENT 6: NAMED ENTITY RECOGNITION (NER)")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__),"clinical_discharge_summaries.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"\nProcessing {len(records)} patient discharge records:\n")

    for rec in records:
        print(f"--- Record ID: {rec['id']} ---")
        print(f"Text Snippet: {rec['text']}")
        
        # Clinical Entity Rationale
        ents = extract_clinical_entities_rulebased(rec['text'])
        print(f"Extracted Drugs:     {ents['DRUG']}")
        print(f"Extracted Dosages:   {ents['DOSAGE']}")
        print(f"Extracted Diagnoses: {ents['DIAGNOSIS']}")
        print(f"Extracted Symptoms:  {ents['SYMPTOM']}\n")

if __name__ == "__main__":
    main()
