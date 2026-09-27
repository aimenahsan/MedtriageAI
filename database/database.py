"""
Database Module for MedTriageAI+
Manages patient records and system data
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import os

# Database path
DB_PATH = 'data/medtriage.db'

def init_database():
    """Initialize SQLite database with required tables"""
    try:
        # Create data folder if it doesn't exist
        Path('data').mkdir(exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Patient records table
        c.execute('''CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            timestamp TEXT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            symptoms TEXT,
            detected_symptoms TEXT,
            ktas_level INTEGER,
            input_method TEXT,
            status TEXT DEFAULT 'pending'
        )''')
        
        # User feedback table
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            patient_id TEXT,
            is_correct INTEGER,
            feedback_text TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )''')
        
        # Model performance table
        c.execute('''CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            metric_name TEXT,
            metric_value REAL
        )''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

def save_patient_record(symptoms, detected_symptoms="", ktas_level=3, input_method="manual", name="Unknown", age=0, gender=""):
    """Save patient record"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        patient_id = f"PT{int(datetime.now().timestamp())}"
        timestamp = datetime.now().isoformat()
        
        c.execute('''INSERT INTO patients 
                    (patient_id, timestamp, name, age, gender, symptoms, 
                     detected_symptoms, ktas_level, input_method, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (patient_id, timestamp, name, age, gender, symptoms, 
                  detected_symptoms, ktas_level, input_method, 'completed'))
        
        conn.commit()
        conn.close()
        
        return patient_id
    except Exception as e:
        print(f"❌ Error saving record: {e}")
        return None

def save_user_feedback(patient_id, is_correct, feedback_text=""):
    """Save user feedback"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        c.execute('''INSERT INTO feedback 
                    (timestamp, patient_id, is_correct, feedback_text)
                    VALUES (?, ?, ?, ?)''',
                 (timestamp, patient_id, int(is_correct), feedback_text))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_session_stats():
    """Get session statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM patients")
        total = c.fetchone()[0] or 0
        
        c.execute("SELECT AVG(ktas_level) FROM patients WHERE ktas_level > 0")
        avg_ktas = c.fetchone()[0] or 3.0
        
        conn.close()
        return {'total': total, 'avg_ktas': avg_ktas, 'avg_conf': 0.85}
    except Exception as e:
        print(f"⚠️ Error getting stats: {e}")
        return {'total': 0, 'avg_ktas': 3.0, 'avg_conf': 0.0}

def get_all_patients():
    """Get all patient records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("SELECT * FROM patients ORDER BY timestamp DESC")
        rows = c.fetchall()
        conn.close()
        
        patients = []
        for row in rows:
            patients.append({'id': row[0], 'patient_id': row[1], 'timestamp': row[2],
                           'name': row[3], 'age': row[4], 'gender': row[5],
                           'symptoms': row[6], 'detected_symptoms': row[7],
                           'ktas_level': row[8], 'input_method': row[9], 'status': row[10]})
        return patients
    except Exception as e:
        return []

def export_to_csv():
    """Export to CSV"""
    try:
        import csv
        patients = get_all_patients()
        if not patients:
            return False
        
        filename = f"medtriage_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=patients[0].keys())
            writer.writeheader()
            writer.writerows(patients)
        return True
    except Exception as e:
        return False
