import os
import sys
import builtins

original_open = builtins.open

KNOWN_DIRS = ['configs', 'models', 'weights', 'input', 'output']


def patched_open(file, *args, **kwargs):
    if isinstance(file, str) and file.startswith('/'):
        for known_dir in KNOWN_DIRS:
            if file.startswith(f'/{known_dir}/'):
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    rel_path = file[1:]
                    new_path = os.path.join(sys._MEIPASS, rel_path)

                    log_path = os.path.join(os.path.expanduser('~'), 'audioseparator_paths.log')
                    with original_open(log_path, 'a') as log:
                        log.write(f"Перенаправление: {file} -> {new_path}\n")
                        log.write(f"Файл существует: {os.path.exists(new_path)}\n\n")

                    file = new_path
                break

    return original_open(file, *args, **kwargs)

builtins.open = patched_open

print("Патч для работы с файлами активирован")