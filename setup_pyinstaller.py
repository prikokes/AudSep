import os
import sys
import re
import subprocess
import shutil
import glob
import torch
import torchaudio
import traceback


def prepare_directories():
    os.makedirs('build', exist_ok=True)
    os.makedirs('dist', exist_ok=True)

    for directory in ['templates', 'models', 'model_loaders', 'utils']:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Auto-generated init file\n")


def fix_code_paths():
    path_patterns = [
        r"(?<!['\"])['\"]/configs/([^'\"]+)['\"]",
        r"(?<!['\"])['\"]/(models|weights|input|output)/([^'\"]+)['\"]",
    ]

    python_files = []
    for root, _, files in os.walk('.'):
        if any(x in root for x in ['.git', 'build', 'dist', 'env', 'venv']):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    for py_file in python_files:
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        modified = False

        if any(re.search(pattern, content) for pattern in path_patterns):
            if 'from utils.path_utils import get_resource_path' not in content:
                import_line = 'from utils.path_utils import get_resource_path\n'
                import_matches = list(re.finditer(r'^import|^from\s+\w+\s+import', content, re.MULTILINE))
                if import_matches:
                    last_import = import_matches[-1]
                    end_of_line = content.find('\n', last_import.start()) + 1
                    content = content[:end_of_line] + import_line + content[end_of_line:]
                else:
                    content = import_line + content
                modified = True

        for pattern in path_patterns:
            def replace_path(match):
                # Конфиги
                if '/configs/' in match.group(0):
                    path = 'configs/' + match.group(1)
                    return f"get_resource_path('{path}')"

                directory = match.group(1)
                filename = match.group(2)
                path = f"{directory}/{filename}"
                return f"get_resource_path('{path}')"

            new_content = re.sub(pattern, replace_path, content)
            if new_content != content:
                content = new_content
                modified = True

        if modified:
            print(f"Исправлены пути в файле: {py_file}")
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(content)


def collect_torch_libraries():
    libraries = []

    torch_dir = os.path.dirname(torch.__file__)
    torch_lib_dir = os.path.join(torch_dir, 'lib')
    if os.path.exists(torch_lib_dir):
        libraries.extend([
            (os.path.join(torch_lib_dir, lib), 'torch/lib')
            for lib in os.listdir(torch_lib_dir) 
            if lib.endswith('.dylib') or lib.endswith('.so')
        ])

    torchaudio_dir = os.path.dirname(torchaudio.__file__)
    torchaudio_lib_dir = os.path.join(torchaudio_dir, 'lib')
    if os.path.exists(torchaudio_lib_dir):
        libraries.extend([
            (os.path.join(torchaudio_lib_dir, lib), 'torchaudio/lib')
            for lib in os.listdir(torchaudio_lib_dir)
            if lib.endswith('.dylib') or lib.endswith('.so')
        ])
    
    return libraries


def check_config_files():
    config_dir = 'configs'
    os.makedirs(config_dir, exist_ok=True)

    config_file = os.path.join(config_dir, 'config_htdemucs_6stems.yaml')
    if not os.path.exists(config_file):
        print(f"Конфигурационный файл {config_file} не найден, создаем заглушку")
        with open(config_file, 'w') as f:
            f.write("""# Конфигурация для модели htdemucs с 6 стемами
model:
  name: htdemucs
  stems: 6
  checkpoint_dir: weights
  sample_rate: 44100

processing:
  batch_size: 1
  shifts: 1
  split: true
  overlap: 0.25
  segment_size: 44100
  device: auto
""")


def build_app():
    try:
        prepare_directories()
        fix_code_paths()
        check_config_files()

        print("Сбор библиотек PyTorch и torchaudio...")
        torch_libs = collect_torch_libraries()
        binary_includes = []
        for src, dst in torch_libs:
            binary_includes.append(f'--add-binary={src}:{dst}')

        for dir_name in ['configs', 'templates', 'models', 'model_loaders', 'utils', 'weights', 'input', 'output']:
            os.makedirs(dir_name, exist_ok=True)

        cmd = [
            'pyinstaller',
            '--name=AudioSeparator',
            '--windowed',
            '--noconfirm',
            '--clean',
            '--add-data=templates:templates',
            '--add-data=models:models',
            '--add-data=model_loaders:model_loaders',
            '--add-data=utils:utils',
            '--add-data=configs:configs',
            '--add-data=weights:weights',
            '--add-data=input:input',
            '--add-data=output:output',
            '--hidden-import=torch',
            '--hidden-import=torchaudio',
            '--hidden-import=templates.audio_separator_app',
            '--hidden-import=PyQt5.QtWidgets',
            '--hidden-import=demucs.htdemucs',
            '--hidden-import=PyQt5.QtCore',
            '--hidden-import=PyQt5.QtGui',
            '--hidden-import=demucs',
            '--hidden-import=soundfile',
            '--hidden-import=customtkinter',
            '--hidden-import=yaml',
            '--hidden-import=utils.path_utils',
        ]

        cmd.extend(binary_includes)

        cmd.append('main.py')

        print("Запуск PyInstaller...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        app_path = 'dist/AudioSeparator.app'
        if os.path.exists(app_path):
            meipass_path = os.path.join(app_path, 'Contents/Resources')
            config_path = os.path.join(meipass_path, 'configs')
            if os.path.exists(config_path):
                configs = os.listdir(config_path)
                print(f"В приложении найдены конфигурационные файлы: {configs}")
            else:
                print("ВНИМАНИЕ: Директория configs не найдена в собранном приложении!")
        
        print("Сборка завершена! Приложение находится в dist/AudioSeparator.app")
        
    except Exception as e:
        print(f"Произошла ошибка при сборке приложения: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    build_app()