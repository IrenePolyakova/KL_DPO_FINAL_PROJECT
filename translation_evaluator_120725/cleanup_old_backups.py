import os 
import time 
from pathlib import Path 
 
def cleanup_old_backups(backup_dir="translation_backups", days_to_keep=7): 
    """Удаляет резервные копии старше указанного количества дней""" 
    if not os.path.exists(backup_dir): 
        return 
 
    current_time = time.time() 
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60) 
 
    for root, dirs, files in os.walk(backup_dir): 
        for file in files: 
            file_path = os.path.join(root, file) 
            if os.path.getmtime(file_path) < cutoff_time: 
                try: 
                    os.remove(file_path) 
                    print(f"Deleted old backup: {file_path}") 
                except: 
                    pass 
 
if __name__ == "__main__": 
    cleanup_old_backups() 
