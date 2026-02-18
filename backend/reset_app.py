import shutil
import sys
from pathlib import Path

# Add project root to path
# Assuming this script is run from project root or backend/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.config import settings
from backend.database import init_db, get_connection
from backend.services.project_service import ProjectService

def reset_app():
    print("⚠️  WARNING: This will delete ALL projects, runs, and data.")
    print("ALL settings will be reset.")
    val = input("Type 'yes' to continue: ")
    if val.lower() != "yes":
        print("Cancelled.")
        return

    # 1. Init DB connection
    print(f"Connecting to database at {settings.data_dir / 'yololabel.db'}...")
    try:
        init_db(settings.data_dir)
    except Exception:
        pass # Might fail if locked, but we try

    # 2. Truncate tables
    try:
        with get_connection() as conn:
            print("Deleting data from database...")
            # Delete in order of dependencies (child first)
            conn.execute("DELETE FROM model_versions")
            conn.execute("DELETE FROM predictions")
            conn.execute("DELETE FROM al_cycles")
            # epoch_metrics and training_runs cascade from projects usually, 
            # but we clear them explicitly to be safe and ensure clean slate
            conn.execute("DELETE FROM epoch_metrics")
            conn.execute("DELETE FROM training_runs")
            conn.execute("DELETE FROM projects")
            conn.execute("DELETE FROM settings") # <--- Added this
            conn.execute("DELETE FROM sqlite_sequence") # Reset ID counters
            conn.commit()
    except Exception as e:
        print(f"Error executing DB commands: {e}")
    
    # 3. Delete folders
    print("Deleting project folders...")
    if settings.projects_dir.exists():
        for item in settings.projects_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    print(f"  - Deleted {item.name}")
                except Exception as e:
                    print(f"  ! Failed to delete {item.name}: {e}")

    # 4. Delete legacy folders (models/, runs/)
    print("Deleting legacy project folders...")
    for folder in ["models", "runs"]:
        path = settings.project_root / folder
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"  - Deleted legacy '{folder}' folder")
            except Exception as e:
                print(f"  ! Failed to delete '{folder}': {e}")

    # 4. Done
    print("\nDone. Please restart the backend server.")

if __name__ == "__main__":
    reset_app()
