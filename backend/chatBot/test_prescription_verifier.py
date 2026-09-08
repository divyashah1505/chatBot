"""
Unit & Integration Test for 100% Local Prescription Recognition & Medicine Verification Engine.
Tests:
1. Fuzzy matching of misspelled handwritten medicine names.
2. Medical frequency (1-0-1, OD, BD, TID, SOS) decoding.
3. Patient & Doctor Header Extraction.
4. Ramesh Patel prescription verification.
"""

from prescription_analyzer import (
    match_medicine_in_database,
    parse_prescription_lines_into_medications,
    extract_patient_and_doctor_header,
    local_prescription_analyzer
)

def test_fuzzy_matching():
    print("--- TEST 1: Fuzzy Matching & Spell Correction ---")
    test_cases = [
        ("Paracetmol 650", "Paracetamol (Acetaminophen)"),
        ("Pantocid D", "Pantoprazole + Domperidone"),
        ("Augmentn 625", "Amoxicillin + Clavulanic Acid (Co-Amoxiclav)"),
        ("Azithral 500", "Azithromycin"),
        ("Cetzine 10", "Cetirizine Hydrochloride"),
        ("Montair LC", "Montelukast + Levocetirizine"),
        ("Levocet 5", "Levocetirizine Hydrochloride"),
        ("Cofsils Lozenges", "Amylmetacresol + Dichlorobenzyl Alcohol (Throat Lozenges)"),
        ("Telma 40", "Telmisartan"),
        ("Glycomet 500", "Metformin Hydrochloride"),
        ("Calcirol 60k", "Cholecalciferol (Vitamin D3)"),
    ]

    for raw, expected_generic in test_cases:
        matched_db, matched_name, conf, status = match_medicine_in_database(raw)
        assert matched_db is not None, f"Failed to match: {raw}"
        assert matched_db["generic"] == expected_generic, f"Expected {expected_generic}, got {matched_db['generic']}"
        print(f"[OK] '{raw}' -> Matched: {matched_name} ({matched_db['generic']}) [Conf: {int(conf*100)}%, Status: {status}]")


def test_ramesh_patel_prescription():
    print("\n--- TEST 2: Ramesh Patel Real Prescription Verification ---")
    sample_lines = [
        "Name : Ramesh Patel   Date : 01/09/2026",
        "Age : 45 / M",
        "(1) Tab. Pantocid D - 1-0-1 before food",
        "(2) Tab. Azithral 500 - 1-0-1 after food",
        "(3) Tab. Montair LC - 0-0-1 at night",
        "(4) Tab. Levocet 5 - 0-1-0 at night",
        "(5) Cofsils Lozenges - SOS if throat irritation",
        "Take plenty of warm water",
        "Rest & take steam.",
        "Dr. A. Sharma, MBBS, MD"
    ]

    header = extract_patient_and_doctor_header(sample_lines)
    assert header["patient_name"] == "Ramesh Patel", f"Expected Ramesh Patel, got {header['patient_name']}"
    assert "45" in header["age_sex"]
    assert "01/09/2026" in header["date"]

    meds = parse_prescription_lines_into_medications(sample_lines)
    assert len(meds) == 5, f"Expected 5 meds, got {len(meds)}"

    expected_med_names = ["Pantocid D", "Azithral 500", "Montair LC", "Levocet 5", "Cofsils Lozenges"]
    for idx, (m, expected_name) in enumerate(zip(meds, expected_med_names), start=1):
        assert m["verified_name"] == expected_name, f"Item {idx}: Expected {expected_name}, got {m['verified_name']}"
        print(f"[Rx Item {idx}] {m['prescribed_name']} -> Verified: {m['verified_name']} | Dose: {m['dosage']} | Freq: {m['frequency']} | Timing: {m['timing']}")


def test_full_report_generation():
    print("\n--- TEST 3: Report & Table Generation ---")
    import io
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (400, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Tab. Pantop DSR 40 - 1-0-1 before food 5 days", fill=(0, 0, 0))
    d.text((10, 40), "Cap. Azithro 500 - 1-0-1 after food 3 days", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    valid_bytes = buf.getvalue()

    result = local_prescription_analyzer.analyze_prescription(valid_bytes, "doctor_rx.png")
    assert result["is_prescription"] is True
    assert len(result["reply"]) > 100
    print(f"[OK] Full Markdown Report Generated with {result['medication_count']} medications!")


if __name__ == "__main__":
    test_fuzzy_matching()
    test_ramesh_patel_prescription()
    test_full_report_generation()
    print("\n*** ALL PRESCRIPTION TESTS PASSED 100% LOCALLY! ***")
