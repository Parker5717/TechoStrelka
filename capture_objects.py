#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   CAPTURE OBJECTS — Приложение для фотографирования объектов        ║
║   для последующего распознавания в сценариях КАСБОТ                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  НАЗНАЧЕНИЕ:                                                         ║
║  • Запуск камеры и отображение в реальном времени                   ║
║  • Фотосъёмка объектов по нажатию клавиши                            ║
║  • Сохранение снимков в организованные папки                         ║
║  • Возможность ввода названия объекта перед съёмкой                 ║
║  • Автоматическая нумерация файлов                                   ║
║                                                                      ║
║  СТРУКТУРА ПАПОК:                                                    ║
║  captured_objects/                                                   ║
║  ├── chair/                                                          ║
║  │   ├── chair_001.jpg                                               ║
║  │   ├── chair_002.jpg                                               ║
║  │   └── ...                                                         ║
║  ├── button/                                                         ║
║  │   └── ...                                                         ║
║  └── ...                                                             ║
║                                                                      ║
║  ЗАПУСК:                                                             ║
║    python capture_objects.py                                         ║
║    python capture_objects.py --cam 1     # другая камера             ║
║    python capture_objects.py --dir ./my_photos  # своя папка         ║
║                                                                      ║
║  УПРАВЛЕНИЕ:                                                         ║
║    N — ввести название нового объекта (создать папку)               ║
║    S — сделать снимок текущего объекта                               ║
║    R — переключиться на другую камеру                                ║
║    Q — выйти                                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════

DEFAULT_CAPTURE_DIR = "captured_objects"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Цвета (BGR)
COLOR_TEXT = (255, 255, 255)
COLOR_HIGHLIGHT = (0, 255, 255)  # жёлтый
COLOR_SUCCESS = (0, 255, 0)      # зелёный
COLOR_ERROR = (0, 0, 255)        # красный
COLOR_BG = (40, 40, 50)          # тёмно-серый


class ObjectCaptureApp:
    """Приложение для захвата и сохранения изображений объектов."""
    
    def __init__(self, capture_dir: str = DEFAULT_CAPTURE_DIR, camera_id: int = 0):
        self.capture_dir = Path(capture_dir)
        self.camera_id = camera_id
        self.current_object = None
        self.object_folders = []
        self.last_capture_time = 0
        self.capture_cooldown = 0.5  # секунд между снимками
        self.message = ""
        self.message_time = 0
        
        # Создаём директорию для сохранений
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        
        # Сканируем существующие папки с объектами
        self.scan_existing_objects()
        
        # Инициализация камеры
        self.cap = None
        self.init_camera()
    
    def scan_existing_objects(self):
        """Сканирует существующие папки с объектами."""
        self.object_folders = []
        for item in self.capture_dir.iterdir():
            if item.is_dir():
                self.object_folders.append(item.name)
        self.object_folders.sort()
    
    def init_camera(self):
        """Инициализирует камеру с заданным ID."""
        if self.cap is not None:
            self.cap.release()
        
        print(f"[INFO] Инициализация камеры {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Не удалось открыть камеру {self.camera_id}")
            return False
        
        # Устанавливаем разрешение (опционально)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"[OK] Камера {self.camera_id} активна")
        return True
    
    def switch_camera(self):
        """Переключается на другую камеру."""
        self.camera_id = 1 - self.camera_id if self.camera_id == 0 else 0
        return self.init_camera()
    
    def get_next_filename(self, object_name: str) -> str:
        """Генерирует следующее имя файла для объекта."""
        object_dir = self.capture_dir / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        
        # Считаем существующие файлы
        existing_files = list(object_dir.glob("*.jpg")) + list(object_dir.glob("*.png"))
        next_num = len(existing_files) + 1
        
        return f"{object_name}_{next_num:03d}.jpg"
    
    def capture_image(self) -> bool:
        """Делает снимок и сохраняет его."""
        current_time = datetime.now().timestamp()
        if current_time - self.last_capture_time < self.capture_cooldown:
            return False
        
        if self.current_object is None:
            self.show_message("Сначала выберите объект (клавиша N)", COLOR_ERROR)
            return False
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.show_message("Ошибка захвата кадра!", COLOR_ERROR)
            return False
        
        # Генерируем имя файла
        filename = self.get_next_filename(self.current_object)
        filepath = self.capture_dir / self.current_object / filename
        
        # Сохраняем изображение
        cv2.imwrite(str(filepath), frame)
        
        self.last_capture_time = current_time
        self.show_message(f"✓ Снимок сохранён: {filename}", COLOR_SUCCESS)
        print(f"[CAPTURE] {filepath}")
        
        return True
    
    def show_message(self, text: str, color: tuple = COLOR_TEXT, duration: float = 3.0):
        """Показывает временное сообщение на экране."""
        self.message = text
        self.message_color = color
        self.message_time = datetime.now().timestamp() + duration
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Рисует интерфейс поверх кадра."""
        h, w = frame.shape[:2]
        
        # Полупрозрачный фон для UI (верхняя часть)
        overlay = frame.copy()
        ui_height = int(h * 0.15)
        cv2.rectangle(overlay, (0, 0), (w, ui_height), COLOR_BG, -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Текущий объект
        y_offset = 40
        if self.current_object:
            text = f"ОБЪЕКТ: {self.current_object.upper()}"
            cv2.putText(frame, text, (20, y_offset), FONT, FONT_SCALE, COLOR_HIGHLIGHT, FONT_THICKNESS)
            
            # Количество снимков этого объекта
            object_dir = self.capture_dir / self.current_object
            if object_dir.exists():
                count = len(list(object_dir.glob("*.jpg")) + list(object_dir.glob("*.png")))
                count_text = f"Снимков: {count}"
                cv2.putText(frame, count_text, (20, y_offset + 35), FONT, 0.6, COLOR_TEXT, 1)
        else:
            text = "ОБЪЕКТ: НЕ ВЫБРАН (нажмите N)"
            cv2.putText(frame, text, (20, y_offset), FONT, FONT_SCALE, COLOR_ERROR, FONT_THICKNESS)
        
        # Список доступных объектов (справа)
        if self.object_folders:
            list_y = 40
            max_display = 5
            for i, obj_name in enumerate(self.object_folders[:max_display]):
                color = COLOR_HIGHLIGHT if obj_name == self.current_object else COLOR_TEXT
                text = f"• {obj_name}"
                cv2.putText(frame, text, (w - 250, list_y + i * 30), FONT, 0.5, color, 1)
            
            if len(self.object_folders) > max_display:
                more_text = f"... ещё {len(self.object_folders) - max_display}"
                cv2.putText(frame, more_text, (w - 250, list_y + max_display * 30), FONT, 0.4, COLOR_TEXT, 1)
        
        # Подсказки по управлению (низ экрана)
        hints_y = h - 60
        hints = [
            "N — новый объект",
            "S — снимок",
            "R — смена камеры",
            "Q — выход"
        ]
        
        # Фон для подсказок
        hints_overlay = frame.copy()
        hints_height = 50
        cv2.rectangle(hints_overlay, (0, h - hints_height), (w, h), (20, 20, 30), -1)
        frame = cv2.addWeighted(hints_overlay, 0.8, frame, 0.2, 0)
        
        for i, hint in enumerate(hints):
            x_pos = 20 + i * (w // 4)
            cv2.putText(frame, hint, (x_pos, hints_y), FONT, 0.5, COLOR_TEXT, 1)
        
        # Сообщение об успехе/ошибке
        if self.message and datetime.now().timestamp() < self.message_time:
            msg_y = ui_height + 50
            cv2.putText(frame, self.message, (20, msg_y), FONT, 0.7, self.message_color, 2)
        
        # Рамка по краям кадра (визуальный акцент)
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), COLOR_TEXT, 1)
        
        return frame
    
    def prompt_object_name(self) -> str:
        """Запрашивает у пользователя название объекта."""
        print("\n" + "="*60)
        print("Введите название объекта (например: button, chair, panel)")
        print("Нажмите Enter для подтверждения или просто Enter для отмены")
        print("="*60)
        
        try:
            name = input("Название объекта: ").strip().lower()
            
            if not name:
                return None
            
            # Очищаем название от недопустимых символов
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            
            if not clean_name:
                print("[ERROR] Недопустимое название!")
                return None
            
            return clean_name
            
        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"[ERROR] Ошибка ввода: {e}")
            return None
    
    def select_object_interactive(self):
        """Интерактивный выбор или создание объекта."""
        self.scan_existing_objects()
        
        if self.object_folders:
            print("\n" + "="*60)
            print("СУЩЕСТВУЮЩИЕ ОБЪЕКТЫ:")
            for i, obj in enumerate(self.object_folders, 1):
                print(f"  {i}. {obj}")
            print(f"  0. Создать новый объект")
            print("="*60)
            
            try:
                choice = input("Выберите номер (или название нового объекта): ").strip()
                
                if choice == "0":
                    new_name = self.prompt_object_name()
                    if new_name:
                        self.current_object = new_name
                        self.object_folders.append(new_name)
                        self.object_folders.sort()
                        self.show_message(f"✓ Объект '{new_name}' создан", COLOR_SUCCESS)
                        print(f"[OK] Создан объект: {new_name}")
                    return
                elif choice.isdigit() and 1 <= int(choice) <= len(self.object_folders):
                    self.current_object = self.object_folders[int(choice) - 1]
                    self.show_message(f"✓ Выбран объект: {self.current_object}", COLOR_SUCCESS)
                    print(f"[OK] Выбран объект: {self.current_object}")
                    return
                elif choice and choice not in ["0"]:
                    # Пользователь ввёл название напрямую
                    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in choice)
                    if clean_name:
                        if clean_name not in self.object_folders:
                            self.object_folders.append(clean_name)
                            self.object_folders.sort()
                        self.current_object = clean_name
                        self.show_message(f"✓ Объект: {clean_name}", COLOR_SUCCESS)
                        print(f"[OK] Выбран объект: {clean_name}")
                        return
                        
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(f"[ERROR] Ошибка выбора: {e}")
        else:
            print("\n[INFO] Нет существующих объектов. Создайте первый!")
            new_name = self.prompt_object_name()
            if new_name:
                self.current_object = new_name
                self.object_folders.append(new_name)
                self.show_message(f"✓ Объект '{new_name}' создан", COLOR_SUCCESS)
                print(f"[OK] Создан объект: {new_name}")
    
    def run(self):
        """Основной цикл приложения."""
        print("\n" + "="*60)
        print("CAPTURE OBJECTS — Фотографирование объектов для КАСБОТ")
        print("="*60)
        print(f"Папка сохранений: {self.capture_dir.absolute()}")
        print("Управление:")
        print("  N — выбрать/создать объект")
        print("  S — сделать снимок")
        print("  R — сменить камеру")
        print("  Q — выход")
        print("="*60 + "\n")
        
        running = True
        
        while running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[ERROR] Не удалось получить кадр. Проверьте камеру.")
                break
            
            # Отрисовка интерфейса
            display_frame = self.draw_ui(frame)
            
            # Показываем окно
            cv2.imshow("Capture Objects - N:объект, S:снимок, R:камера, Q:выход", display_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q или ESC
                running = False
                print("\n[INFO] Завершение работы...")
            
            elif key == ord('n') or key == ord('N'):
                print("\n[INFO] Выбор объекта...")
                self.select_object_interactive()
            
            elif key == ord('s') or key == ord('S'):
                self.capture_image()
            
            elif key == ord('r') or key == ord('R'):
                print("\n[INFO] Переключение камеры...")
                if self.switch_camera():
                    self.show_message("✓ Камера переключена", COLOR_SUCCESS)
                else:
                    self.show_message("✗ Ошибка переключения камеры", COLOR_ERROR)
        
        # Освобождение ресурсов
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("СЕАНС ЗАВЕРШЁН")
        print(f"Всего объектов: {len(self.object_folders)}")
        if self.object_folders:
            print("Объекты:", ", ".join(self.object_folders))
        print(f"Папка: {self.capture_dir.absolute()}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CAPTURE OBJECTS — Фотографирование объектов для последующего распознавания",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python capture_objects.py                      # запуск с камерой по умолчанию
  python capture_objects.py --cam 1              # использовать вторую камеру
  python capture_objects.py --dir ./my_objects   # сохранить в свою папку
        """
    )
    
    parser.add_argument(
        "--cam", "--camera",
        type=int,
        default=0,
        help="ID камеры (по умолчанию: 0)"
    )
    
    parser.add_argument(
        "--dir", "--directory",
        type=str,
        default=DEFAULT_CAPTURE_DIR,
        help=f"Папка для сохранения снимков (по умолчанию: {DEFAULT_CAPTURE_DIR})"
    )
    
    args = parser.parse_args()
    
    try:
        app = ObjectCaptureApp(capture_dir=args.dir, camera_id=args.cam)
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
