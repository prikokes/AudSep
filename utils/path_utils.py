import os
import sys


def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    if relative_path.startswith('/'):
        relative_path = relative_path[1:]

    full_path = os.path.join(base_path, relative_path)

    debug_log = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'debug_paths.log')
    with open(debug_log, 'a') as f:
        f.write(f"Запрошен ресурс: {relative_path}\n")
        f.write(f"Полный путь: {full_path}\n")
        f.write(f"Существует: {os.path.exists(full_path)}\n\n")
    
    return full_path


def ensure_dir_exists(directory):
    os.makedirs(directory, exist_ok=True)
