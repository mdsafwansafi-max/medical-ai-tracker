import os
import json
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image

app = FastAPI(title="Medical AI Journey API")

# CORS سیٹ اپ تاکہ کسی بھی براؤزر سے ریکویسٹ وصول ہو سکے
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render کے Environment Variables سے API Key حاصل کرے گا
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are an expert Medical OCR and Data Extraction Assistant. Analyze the uploaded medical document image and extract all structured data accurately.

STRICT INSTRUCTIONS:
1. Classify the document into one of the following exact types:
   - "CHECKUP_NOTE" (Doctor's clinical notes, symptoms, physical examination, vitals)
   - "TEST_REPORT" (Lab investigations, Blood tests, Radiology, Urine tests, etc.)
   - "PRESCRIPTION" (Rx slips with listed medications and instructions)
   - "OTHER" (Billing receipts, generic ID cards, non-medical text)

2. Transcribe handwritten notes accurately. If handwriting is illegible, mark that field as null or "Unclear".
3. Extract date, doctor/clinic name, patient details, and findings.
4. Always respond ONLY in valid, strictly parsable JSON format without Markdown code blocks or wrapping commentary.

JSON OUTPUT STRUCTURE:
{
  "document_type": "CHECKUP_NOTE" | "TEST_REPORT" | "PRESCRIPTION" | "OTHER",
  "document_date": "YYYY-MM-DD or null",
  "doctor": {
    "name": "Doctor Name or null",
    "clinic_or_hospital": "Clinic Name or null"
  },
  "patient_name_on_doc": "Patient Name or null",
  "diagnosis_or_symptoms": "Summary of diagnosis or symptoms",
  "prescribed_medicines": [
    {
      "name": "Medicine Name",
      "dosage": "Dosage (e.g. 500mg, 1 tablet)",
      "frequency": "Frequency (e.g. 1-0-1, Twice daily)",
      "duration": "Duration (e.g. 5 days)",
      "instructions": "Instructions (e.g. After meals)"
    }
  ],
  "lab_test_results": [
    {
      "test_name": "Test Name (e.g. Complete Blood Count)",
      "parameter": "Parameter Name (e.g. Hemoglobin)",
      "value": 12.5,
      "unit": "g/dL",
      "normal_range": "13.5 - 17.5",
      "is_abnormal": true
    }
  ],
  "confidence_score": "HIGH" | "MEDIUM" | "LOW"
}
"""

@app.get("/")
def home():
    return {"status": "Medical AI Backend is Running Successfully"}

@app.post("/api/process-medical-doc")
async def process_medical_document(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        response = model.generate_content([SYSTEM_PROMPT, image])
        extracted_data = json.loads(response.text)

        return {
            "status": "success",
            "filename": file.filename,
            "data": extracted_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
