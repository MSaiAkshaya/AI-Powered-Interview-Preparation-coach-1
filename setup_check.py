"""
setup_check.py
--------------
Run this BEFORE `streamlit run app.py` to confirm:
1. Python version is OK
2. All required packages import successfully
3. .env file exists with IBM Cloud / watsonx.ai credentials set

Usage:
    python setup_check.py
"""

import sys
import os
import importlib

print("=" * 60)
print("AI Interview Coach (IBM Cloud Edition) — Setup Check")
print("=" * 60)

# 1. Python version
print(f"\n[1/3] Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 9):
    print("    WARNING: Python 3.9+ recommended.")
else:
    print("    OK")

# 2. Package imports
print("\n[2/3] Checking required packages...")
packages = {
    "streamlit": "streamlit",
    "ibm_watsonx_ai": "ibm-watsonx-ai",
    "pdfplumber": "pdfplumber",
    "dotenv": "python-dotenv",
}

missing = []
for module_name, pip_name in packages.items():
    try:
        importlib.import_module(module_name)
        print(f"    OK   - {pip_name}")
    except ImportError:
        print(f"    MISSING - {pip_name}")
        missing.append(pip_name)

if missing:
    print(f"\n  Run this to install missing packages:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("\n  All packages installed correctly.")

# 3. .env / IBM Cloud credentials check
print("\n[3/3] Checking .env file for IBM Cloud credentials...")
if not os.path.exists(".env"):
    print("    MISSING - .env file not found in this folder.")
    print("    Run: cp .env.example .env   (then edit it with your IBM Cloud credentials)")
else:
    from dotenv import load_dotenv
    load_dotenv()

    checks = {
        "WATSONX_API_KEY": "IBM Cloud API key (watsonx.ai)",
        "WATSONX_PROJECT_ID": "watsonx.ai IBM Cloud Project ID",
        "WATSONX_URL": "watsonx.ai region URL",
        "GRANITE_MODEL_ID": "IBM Granite Foundation Model ID",
    }
    all_ok = True
    for var, label in checks.items():
        value = os.getenv(var)
        placeholder = value and "your_" in value.lower()
        if not value or placeholder:
            print(f"    MISSING - {var} ({label}) is not set properly.")
            all_ok = False
        else:
            shown = value if len(value) < 12 else f"{value[:8]}...hidden"
            print(f"    OK   - {var} is set ({shown})")

print("\n" + "=" * 60)
if not missing and os.path.exists(".env"):
    print("Setup looks good. Run: streamlit run app.py")
else:
    print("Fix the items marked above, then re-run this script.")
print("=" * 60)
