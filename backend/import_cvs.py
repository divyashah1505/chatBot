"""
Batch CV Importer Script.
Run this script to import CV files (.pdf, .txt) from a 'resumes/' folder directly into MongoDB.

Usage:
    python import_cvs.py
"""

import os
import glob
from chatBot.resume_parser import process_and_save_cv
from db.models import get_all_candidates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUMES_DIR = os.path.join(BASE_DIR, "resumes")


def import_all_resumes():
    if not os.path.exists(RESUMES_DIR):
        os.makedirs(RESUMES_DIR, exist_ok=True)
        print(f"Created '{RESUMES_DIR}' folder. Please place .pdf or .txt CV files there and run this script again.")
        return

    files = glob.glob(os.path.join(RESUMES_DIR, "*.*"))
    supported_files = [f for f in files if f.lower().endswith((".pdf", ".txt"))]

    if not supported_files:
        print(f"No .pdf or .txt CV files found in '{RESUMES_DIR}'.")
        return

    print(f"Found {len(supported_files)} CV file(s) to import...")

    imported_count = 0
    for file_path in supported_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            cand = process_and_save_cv(file_bytes, filename)
            print(f"✅ Imported: {cand['name']} | Role: {cand['title']} | Exp: {cand['experience_years']} Yrs")
            imported_count += 1
        except Exception as e:
            print(f"❌ Error importing '{filename}': {e}")

    print(f"\nTotal successfully imported: {imported_count}")
    print(f"Total candidates in database now: {len(get_all_candidates())}")


if __name__ == "__main__":
    import_all_resumes()
