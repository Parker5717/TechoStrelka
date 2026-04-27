#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Приложение для захвата изображений объектов для обучения модели распознавания.
Использует системное диалоговое окно для ввода названия (полная поддержка кириллицы).
Отображает текст на кадре через PIL.
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
import subprocess

# Попытка импортировать библиотеку для транслитерации
try:
    from transliterate import translit
    HAS_TRANSLIT = True
except ImportError:
    HAS_TRANSLIT = False

def simple_translit(text):
    """Упрощенная транслитерация кириллицы в латиницу."""
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo', 'ж': 'zh',
        'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
        'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu',
        'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh',
        'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O',
        'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts',
        'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu',
        'Я': 'Ya'
    }
    return ''.join(mapping.get(char, char) for char in text)

def get_latin_name(name):
    """Преобразует имя в латиницу."""
    if HAS_TRANSLIT:
        try:
            return translit(name, 'ru', reversed=True)
        except:
            pass
    return simple_translit(name)

def get_system_input(prompt_title="Ввод", prompt_text="Введите название объекта:"):
    """
    Открывает системное диалоговое окно для ввода текста.
    Поддерживает кириллицу нативно.
    Пытается использовать tkinter, если нет - zenity (Linux) или osascript (Mac).
    """
    # Попытка 1: Tkinter (работает везде, где есть Python GUI)
    try:
        import tkinter as tk
        from tkinter import simpledialog
        
        root = tk.Tk()
        root.withdraw() # Скрыть главное окно
        root.attributes('-topmost', True) # Поверх всех окон
        
        result = simpledialog.askstring(prompt_title, prompt_text, parent=root)
        root.destroy()
        return result
    except Exception as e:
        print(f"[WARN] Tkinter не доступен ({e}), пробуем другие методы...")

    # Попытка 2: Zenity (Linux GNOME)
    try:
        cmd = ['zenity', '--entry', '--title', prompt_title, '--text', prompt_text]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Попытка 3: Osascript (macOS)
    try:
        script = f'display dialog "{prompt_text}" default answer "" with title "{prompt_title}"\nreturn text returned of result'
        cmd = ['osascript', '-e', script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    print("[ERROR] Не удалось открыть диалоговое окно ввода.")
    print("       Убедитесь, что установлен tkinter: sudo apt-get install python3-tk")
    return None

def draw_text_pil(image, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
    """Рисует текст с поддержкой UTF-8 через PIL."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Конвертация BGR -> RGB для PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Подбор шрифта
    font_size = max(12, int(24 * font_scale))
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "arial.ttf"
    ]
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except:
            continue
    
    if not font:
        try:
            font = ImageFont.load_default()
        except:
            # Если совсем ничего нет, рисуем квадрат вместо текста
            h, w = image.shape[:2]
            cv2.rectangle(image, (position[0], position[1]-20), (position[0]+200, position[1]+20), (0,0,0), -1)
            cv2.putText(image, "NO FONT", (position[0], position[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            return image

    # Отрисовка фона и текста
    bbox = draw.textbbox(position, text, font=font)
    # Рисуем полупрозрачный фон
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(bbox, fill=(0, 0, 0, 180))
    
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    
    # Конвертация обратно в BGR для OpenCV
    return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

class ObjectCaptureApp:
    def __init__(self, camera_id=0, save_dir="captured_objects"):
        self.camera_id = camera_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Не удалось открыть камеру {camera_id}")
            sys.exit(1)
            
        self.current_object = None
        self.existing_objects = self.scan_existing_objects()
        
        print("[INFO] Приложение запущено.")
        print("[INFO] Клавиши:")
        print("  - 'N': Новый объект (откроется окно ввода)")
        print("  - 'S': Сделать снимок")
        print("  - 'Q' или 'Esc': Выход")

    def scan_existing_objects(self):
        objects = []
        if self.save_dir.exists():
            for item in self.save_dir.iterdir():
                if item.is_dir():
                    objects.append(item.name)
        return objects

    def get_next_filename(self, object_name):
        obj_dir = self.save_dir / object_name
        obj_dir.mkdir(exist_ok=True)
        existing_files = list(obj_dir.glob("*.jpg"))
        next_num = len(existing_files) + 1
        return obj_dir / f"{object_name}_{next_num:03d}.jpg"

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            h, w = frame.shape[:2]
            
            # Подготовка текста статуса
            status_lines = []
            color = (200, 200, 200)
            
            if self.current_object:
                status_lines.append(f"Объект: {self.current_object}")
                status_lines.append("Нажмите 'S' для снимка, 'N' для нового")
                color = (0, 255, 0)
            else:
                status_lines.append("Объект не выбран")
                status_lines.append("Нажмите 'N' для создания нового объекта")
                color = (0, 100, 255)

            # Отрисовка фона под текстом
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            
            # Отрисовка текста с поддержкой кириллицы
            y_start = 25
            line_height = 28
            for i, line in enumerate(status_lines):
                y_pos = y_start + i * line_height
                frame = draw_text_pil(frame, line, (15, y_pos), font_scale=0.8, color=color)

            # Инфо о количестве объектов
            if self.existing_objects:
                info = f"Доступно объектов: {len(self.existing_objects)}"
                frame = draw_text_pil(frame, info, (w - 250, 25), font_scale=0.6, color=(150, 150, 150))

            cv2.imshow('CASBOT Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('n') or key == ord('N'):
                # Вызов системного диалога
                user_input = get_system_input("Новый объект", "Введите название (можно по-русски):")
                
                if user_input and user_input.strip():
                    raw_name = user_input.strip()
                    latin_name = get_latin_name(raw_name)
                    safe_name = "".join(c for c in latin_name if c.isalnum() or c in ('-', '_'))
                    
                    if safe_name:
                        self.current_object = safe_name
                        if safe_name not in self.existing_objects:
                            self.existing_objects.append(safe_name)
                        print(f"[OK] Объект: '{raw_name}' -> папка '{safe_name}'")
                    else:
                        print("[ERR] Недопустимое имя.")
                else:
                    print("[INFO] Ввод отменен.")
                    
            elif key == ord('s') or key == ord('S'):
                if self.current_object:
                    filepath = self.get_next_filename(self.current_object)
                    cv2.imwrite(str(filepath), frame)
                    print(f"[SNAP] Сохранено: {filepath}")
                    
                    # Эффект вспышки
                    flash = frame.copy()
                    cv2.rectangle(flash, (0,0), (w,h), (255,255,255), -1)
                    cv2.addWeighted(flash, 0.4, frame, 0.6, 0, frame)
                    cv2.imshow('CASBOT Capture', frame)
                    cv2.waitKey(50)
                else:
                    print("[WARN] Сначала выберите объект (N)!")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Захват объектов CASBOT")
    parser.add_argument("--cam", type=int, default=0, help="ID камеры")
    parser.add_argument("--dir", type=str, default="captured_objects", help="Папка сохранения")
    args = parser.parse_args()
    
    # Проверка зависимостей
    try:
        import tkinter
    except ImportError:
        print("[CRITICAL] Не найден модуль tkinter! Без него ввод кириллицы невозможен.")
        print("Установите его: sudo apt-get install python3-tk (Ubuntu/Debian)")
        sys.exit(1)

    app = ObjectCaptureApp(camera_id=args.cam, save_dir=args.dir)
    app.run()
