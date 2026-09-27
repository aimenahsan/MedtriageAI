"""
Database package initialization
Re-exports all database functions for easy importing

CHANGE NOTE: this previously re-exported get_patient_record and get_feedback,
which don't exist in database.py (that module has get_all_patients and
export_to_csv instead) -- `import database` / `from database import ...`
would have raised ImportError immediately. app_production.py sidesteps this
package entirely via importlib.util.spec_from_file_location(), so it never
surfaced there, but it would break for anyone importing this the normal way.
"""

from .database import (
    init_database,
    save_patient_record,
    save_user_feedback,
    get_session_stats,
    get_all_patients,
    export_to_csv,
    DB_PATH
)

__all__ = [
    'init_database',
    'save_patient_record',
    'save_user_feedback',
    'get_session_stats',
    'get_all_patients',
    'export_to_csv',
    'DB_PATH'
]
