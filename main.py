#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║   КАСБОТ v1.0 — ИИ-наставник первого дня на заводе                  ║
║   Technostrelka 2026 | Трек: Компьютерное зрение | Кейс: Kaspersky  ║
╠══════════════════════════════════════════════════════════════════════╣
║  ПРОБЛЕМА:  Новичок не знает куда идти, что делать, чего бояться.   ║
║  РЕШЕНИЕ:   Смартфон = AR-наставник. Камера видит → система ведёт.  ║
║  ОРИГИНАЛ:  НЕ просто инструкция. Это КВЕСТ с очками безопасности,  ║
║             AR-подсветкой действий и эмоциональной обратной связью.  ║
╚══════════════════════════════════════════════════════════════════════╝

ПОЧЕМУ ЭТО РАБОТАЕТ ДЛЯ МОЛОДЁЖИ:
  • Геймификация (Safety Score 0–100) — язык, понятный Z-поколению
  • Мгновенная обратная связь через камеру — нет ожидания наставника
  • AR-стрелки и подсветка кнопок — как в видеоиграх
  • Нет страха ошибки: система скажет что не так и что исправить
  • Работает на обычном смартфоне, ноутбуке — ничего не нужно покупать

АРХИТЕКТУРА (1 файл = 3 независимых слоя):
  ┌──────────────────────────────────────────────────────┐
  │ CV-СЛОЙ    YOLOv8n (персона, объекты)                │
  │            + HSV-анализ (цвет каски/жилета)          │
  │            + MediaPipe Hands (трекинг рук)           │
  ├──────────────────────────────────────────────────────┤
  │ FSM-СЛОЙ   5 состояний с буфером подтверждения       │
  │            PPE_CHECK → ACCESS → STEP1 → STEP2 → DONE │
  ├──────────────────────────────────────────────────────┤
  │ UI-СЛОЙ    OpenCV overlay: Safety Score, AR-стрелки, │
  │            статусная строка, баннеры состояний        │
  └──────────────────────────────────────────────────────┘

ЗАПУСК:
  pip install ultralytics opencv-python numpy mediapipe
  python main.py              # обычный режим
  python main.py --demo       # демо-режим (клавиши управления)
  python main.py --cam 1      # другой индекс камеры

КЛАВИШИ (в --demo режиме для жюри):
  H — каска обнаружена (симуляция)
  V — жилет обнаружен  (симуляция)
  SPACE — перейти на следующий шаг
  R — сбросить и начать заново
  Q — выйти

ВЕСА МОДЕЛИ: yolov8n.pt (~6MB, скачивается автоматически при первом запуске)
Источник: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
"""

import cv2
import numpy as np
import sys
import time
import argparse
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from PIL import ImageFont, ImageDraw, Image

# ══════════════════════════════════════════════════════════════════
#  ЗАВИСИМОСТИ (с graceful fallback — прототип не упадёт без них)
# ══════════════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[!] YOLO недоступен. Установите: pip install ultralytics")

# MediaPipe — для детекции рук (опционально, но очень эффективно)
try:
    import mediapipe as mp
    # Совместимость с разными версиями MediaPipe
    if hasattr(mp, 'solutions'):
        _mp_hands_solution = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
    else:
        # Для новых версий MediaPipe (0.12+)
        from mediapipe.solutions import hands as _mp_hands_solution
        from mediapipe.solutions import drawing_utils as _mp_drawing
    MEDIAPIPE_AVAILABLE = True
    print("[OK] MediaPipe подключён — трекинг рук активен")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[INFO] MediaPipe не найден — трекинг рук отключён")
    print("       Установите: pip install mediapipe")
except AttributeError:
    MEDIAPIPE_AVAILABLE = False
    print("[INFO] MediaPipe установлен, но версия несовместима — трекинг рук отключён")
    print("       Попробуйте обновить: pip install --upgrade mediapipe")


# ══════════════════════════════════════════════════════════════════
#  КОНФИГУРАЦИЯ — цвета, пороги, HSV-диапазоны
# ══════════════════════════════════════════════════════════════════

# Цветовая палитра UI (BGR для OpenCV)
C = {
    "green":    (50,  200,  50),
    "red":      (40,   40, 220),
    "yellow":   (0,   210, 210),
    "orange":   (0,   140, 255),
    "blue":     (220, 100,  40),
    "cyan":     (200, 200,  40),
    "purple":   (180,  50, 180),
    "white":    (235, 235, 235),
    "black":    (15,   15,  15),
    "dark_bg":  (28,   28,  38),
    "mid_gray": (90,   90,  90),
}

# HSV-диапазоны цветов СИЗ
# Каска бывает жёлтой, оранжевой, белой — учитываем все варианты
HSV_RANGES = {
    "yellow": (np.array([18, 100, 120]), np.array([38, 255, 255])),
    "orange": (np.array([5,  120, 120]), np.array([18, 255, 255])),
    "white":  (np.array([0,    0, 200]), np.array([180, 35, 255])),
    "lime":   (np.array([38,  80, 100]), np.array([75, 255, 255])),  # зелёный жилет
}

# COCO-классы (YOLOv8n стандартные веса)
COCO_PERSON = 0
# Объекты, которые "сходят" за оборудование в демо-условиях
COCO_EQUIPMENT = {
    56: "рабочее место",  # chair
    57: "рабочее место",  # couch
    58: "рабочее место",  # potted plant
    60: "рабочее место",  # dining table
    62: "рабочее место",  # tv
    63: "пульт",          # laptop
    64: "компонент",      # mouse
    66: "инструмент",     # keyboard
    73: "панель",         # book
    39: "оборудование",   # bottle (любой цилиндрический объект)
    76: "инструмент",     # scissors
}

CONFIRM_FRAMES  = 28    # кадров для уверенной детекции (≈1 сек при 30fps)
MIN_STATE_TIME  = 1.5   # секунд — минимум в состоянии до перехода
YOLO_INTERVAL   = 3     # каждые N кадров запускаем YOLO (экономия CPU)


# ══════════════════════════════════════════════════════════════════
#  FSM — КОНЕЧНЫЙ АВТОМАТ СОСТОЯНИЙ
# ══════════════════════════════════════════════════════════════════

class State(Enum):
    PPE_CHECK         = 0  # Проверка средств защиты (каска + жилет)
    ACCESS_GRANTED    = 1  # Допуск выдан — анимация + Safety Score +40
    STEP1_WORKSTATION = 2  # Найти рабочее место — навести камеру
    STEP2_POWER       = 3  # Включить питание — поднять руку к кнопке
    COMPLETE          = 4  # Квест пройден — сертификат первого дня

# Метаданные каждого состояния — заголовок, подсказка, цвет, очки
STATE_META = {
    State.PPE_CHECK: {
        "title":  "ПРОВЕРКА СРЕДСТВ ЗАЩИТЫ",
        "hint":   "Убедитесь: каска и жилет видны в камере",
        "color":  C["yellow"],
        "points": 0,
    },
    State.ACCESS_GRANTED: {
        "title":  "ДОПУСК К РАБОТЕ ВЫДАН!",
        "hint":   "Отлично! Вы защищены. Идём к рабочему месту...",
        "color":  C["green"],
        "points": 40,
    },
    State.STEP1_WORKSTATION: {
        "title":  "ШАГ 1 / 2  —  НАЙДИТЕ РАБОЧЕЕ МЕСТО",
        "hint":   "Наведите камеру на стол или панель управления",
        "color":  C["cyan"],
        "points": 20,
    },
    State.STEP2_POWER: {
        "title":  "ШАГ 2 / 2  —  ВКЛЮЧИТЕ ПИТАНИЕ",
        "hint":   "Поднимите руку к кнопке ПУСК — фиксируем жест",
        "color":  C["orange"],
        "points": 30,
    },
    State.COMPLETE: {
        "title":  "ПЕРВЫЙ ДЕНЬ ПРОЙДЕН!",
        "hint":   "Вы официально допущены к самостоятельной работе.",
        "color":  C["purple"],
        "points": 10,
    },
}

NEXT_STATE = {
    State.PPE_CHECK:         State.ACCESS_GRANTED,
    State.ACCESS_GRANTED:    State.STEP1_WORKSTATION,
    State.STEP1_WORKSTATION: State.STEP2_POWER,
    State.STEP2_POWER:       State.COMPLETE,
}


# ══════════════════════════════════════════════════════════════════
#  DATACLASS — ГЛОБАЛЬНОЕ СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# ══════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    """Единственный объект, который хранит всё состояние системы."""

    current: State    = State.PPE_CHECK
    score:   int      = 0

    # Флаги СИЗ (результат после CONFIRM_FRAMES подтверждений)
    helmet_ok: bool = False
    vest_ok:   bool = False

    # Счётчики стабильной детекции (сбрасываются при потере объекта)
    helmet_cnt:   int = 0
    vest_cnt:     int = 0
    hand_cnt:     int = 0
    target_cnt:   int = 0

    # Симуляция для демо-режима (нажатые клавиши H / V)
    sim_helmet: bool = False
    sim_vest:   bool = False

    # Временные метки
    state_since:  float = field(default_factory=time.time)
    last_trans:   float = field(default_factory=time.time)

    # Фаза пульсации (для анимаций)
    phase: float = 0.0

    # ── Методы ──────────────────────────────────────────────────

    def time_in_state(self) -> float:
        return time.time() - self.state_since

    def try_advance(self, target: State) -> bool:
        """Переход с проверкой минимального времени в состоянии."""
        if self.time_in_state() < MIN_STATE_TIME:
            return False
        self.score = min(100, self.score + STATE_META[target]["points"])
        self.current = target
        self.state_since = time.time()
        self.last_trans  = time.time()
        # Сбрасываем счётчики при переходе
        self.hand_cnt    = 0
        self.target_cnt  = 0
        return True


# ══════════════════════════════════════════════════════════════════
#  CV-СЛОЙ:  ДЕТЕКЦИЯ
# ══════════════════════════════════════════════════════════════════

def hsv_coverage(zone: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Доля пикселей zone, попадающих в HSV-диапазон [lo..hi]."""
    if zone.size < 100:
        return 0.0
    hsv  = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    return np.count_nonzero(mask) / mask.size


def detect_ppe_colors(frame: np.ndarray, person_box: Optional[Tuple]) -> Dict[str, bool]:
    """
    Детектирует СИЗ через анализ HSV-цветов внутри зоны человека.

    Зачем цветовой анализ, а не нейросеть?
    ─────────────────────────────────────────
    Заводские каски и жилеты — ярко-жёлтые/оранжевые намеренно:
    это требование ГОСТ для видимости. HSV-анализ именно таких цветов
    работает надёжно и быстро без специальных весов модели.
    Метод: верхняя треть bbox → зона каски; середина → зона жилета.
    """
    h, w = frame.shape[:2]

    if person_box:
        x1, y1, x2, y2 = (int(v) for v in person_box)
        ph = y2 - y1
        # Зона каски — верхняя 30% bbox человека
        zone_helmet = frame[y1 : y1 + ph * 3 // 10, x1:x2]
        # Зона жилета — центральная 50% по высоте bbox
        zone_vest   = frame[y1 + ph // 4 : y1 + ph * 3 // 4, x1:x2]
    else:
        # Без персоны — ищем в глобальных зонах (удобно для демо)
        zone_helmet = frame[: h // 4, :]
        zone_vest   = frame[h // 5 : 3 * h // 4, :]

    THRESH_HELMET = 0.04   # 4% пикселей зоны — достаточно
    THRESH_VEST   = 0.06   # жилет должен занимать больше площади

    # Каска: жёлтая, оранжевая ИЛИ белая (разные стандарты на заводах)
    helmet = (
        hsv_coverage(zone_helmet, *HSV_RANGES["yellow"]) > THRESH_HELMET or
        hsv_coverage(zone_helmet, *HSV_RANGES["orange"]) > THRESH_HELMET or
        hsv_coverage(zone_helmet, *HSV_RANGES["white"])  > THRESH_HELMET * 2
    )

    # Жилет: жёлтый, оранжевый или салатовый (все варианты рабочих жилетов)
    vest = (
        hsv_coverage(zone_vest, *HSV_RANGES["yellow"]) > THRESH_VEST or
        hsv_coverage(zone_vest, *HSV_RANGES["orange"]) > THRESH_VEST or
        hsv_coverage(zone_vest, *HSV_RANGES["lime"])   > THRESH_VEST
    )

    return {"helmet": helmet, "vest": vest}


def get_largest_person(results) -> Optional[Tuple]:
    """Возвращает xyxy-бокс наибольшего человека из YOLO-результатов."""
    if results is None:
        return None
    best_area, best_box = 0.0, None
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls[0]) != COCO_PERSON:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box  = (x1, y1, x2, y2)
    return best_box


def detect_equipment(results, fh: int, fw: int) -> bool:
    """
    Определяет, что камера направлена на рабочее оборудование.
    Логика: ищем объект из COCO_EQUIPMENT ИЛИ любой достаточно
    крупный объект в нижней половине кадра (рабочая поверхность).
    """
    if results is None:
        return False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            if cls in COCO_EQUIPMENT:
                return True
            # Fallback: крупный объект в нижней части кадра = стол/стенд
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if y1 > fh * 0.35:
                if (x2 - x1) * (y2 - y1) > fh * fw * 0.06:
                    return True
    return False


# ══════════════════════════════════════════════════════════════════
#  CV-СЛОЙ:  МЕДИАПАЙП (РУКИ)
# ══════════════════════════════════════════════════════════════════

_mp_hands_instance = None  # ленивая инициализация

def get_mp_hands():
    """Возвращает (и создаёт при первом вызове) экземпляр Hands."""
    global _mp_hands_instance
    if not MEDIAPIPE_AVAILABLE:
        return None
    if _mp_hands_instance is None:
        _mp_hands_instance = _mp_hands_solution.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.45,
        )
    return _mp_hands_instance


def detect_raised_hand(frame: np.ndarray) -> bool:
    """
    Возвращает True, если рука обнаружена в верхней половине кадра.

    Почему именно "поднятая рука"?
    ─────────────────────────────────────────────────────────────
    Жест "тянусь к кнопке ПУСК" — самый естественный первый шаг
    при запуске оборудования. Система распознаёт намерение, а не
    просто объект. Это делает интерфейс живым и интуитивным для
    молодого сотрудника, привыкшего к сенсорным экранам.
    """
    hands = get_mp_hands()
    if hands is None:
        return False
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if not result.multi_hand_landmarks:
            return False
        h = frame.shape[0]
        for lm in result.multi_hand_landmarks:
            # landmark[0] = запястье; если выше 60% высоты — рука поднята
            if lm.landmark[0].y < 0.60:
                return True
    except Exception:
        pass
    return False


def draw_hand_landmarks(frame: np.ndarray):
    """Рисуем скелет руки (для наглядности на демо)."""
    if not MEDIAPIPE_AVAILABLE:
        return
    hands = get_mp_hands()
    if hands is None:
        return
    try:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands.process(rgb)
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    frame, lm,
                    _mp_hands_solution.HAND_CONNECTIONS,
                    _mp_drawing.DrawingSpec(color=C["cyan"], thickness=2, circle_radius=3),
                    _mp_drawing.DrawingSpec(color=C["white"], thickness=1),
                )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  FSM-СЛОЙ:  ОБНОВЛЕНИЕ СОСТОЯНИЯ
# ══════════════════════════════════════════════════════════════════

def update_fsm(st: AppState, ppe: Dict, hand_up: bool, target_found: bool):
    """
    Центральная логика FSM. Вызывается каждый кадр.

    Принцип стабильности:
    ─────────────────────────────────────────────────────────────
    Каждый переход требует CONFIRM_FRAMES (~28) кадров
    непрерывной детекции. Это ≈1 секунда при 28fps.
    Такой буфер защищает от ложных срабатываний на одиночные шумы
    детекции, что критично при демонстрации жюри.
    """

    if st.current == State.PPE_CHECK:
        # ── Счётчики СИЗ: растём при детекции, сбрасываемся без неё ──
        h_seen = ppe.get("helmet", False) or st.sim_helmet
        v_seen = ppe.get("vest",   False) or st.sim_vest

        st.helmet_cnt = st.helmet_cnt + 1 if h_seen else max(0, st.helmet_cnt - 2)
        st.vest_cnt   = st.vest_cnt   + 1 if v_seen else max(0, st.vest_cnt   - 2)

        st.helmet_ok = st.helmet_cnt >= CONFIRM_FRAMES
        st.vest_ok   = st.vest_cnt   >= CONFIRM_FRAMES

        # Оба СИЗ подтверждены → выдаём допуск
        if st.helmet_ok and st.vest_ok:
            st.try_advance(State.ACCESS_GRANTED)

    elif st.current == State.ACCESS_GRANTED:
        # Автоматический переход на шаг 1 через ~2.5 сек
        if st.time_in_state() >= 2.5:
            st.try_advance(State.STEP1_WORKSTATION)

    elif st.current == State.STEP1_WORKSTATION:
        # Ищем рабочее место в кадре
        found = target_found
        st.target_cnt = st.target_cnt + 1 if found else max(0, st.target_cnt - 1)
        if st.target_cnt >= CONFIRM_FRAMES:
            st.try_advance(State.STEP2_POWER)

    elif st.current == State.STEP2_POWER:
        # Ждём поднятой руки (тянется к кнопке)
        st.hand_cnt = st.hand_cnt + 1 if hand_up else max(0, st.hand_cnt - 1)
        if st.hand_cnt >= CONFIRM_FRAMES:
            st.try_advance(State.COMPLETE)

    # State.COMPLETE — терминальное, переходов нет


# ══════════════════════════════════════════════════════════════════
#  UI-СЛОЙ:  ФУНКЦИИ РЕНДЕРИНГА
# ══════════════════════════════════════════════════════════════════

def alpha_overlay(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  color: Tuple, alpha: float = 0.78):
    """Полупрозрачный прямоугольник — базовый блок UI."""
    roi = frame[y1:y2, x1:x2]
    rect = np.full_like(roi, color[::-1] if len(color) == 3 else color, dtype=np.uint8)
    # color приходит в BGR
    rect[:, :, 0] = color[0]
    rect[:, :, 1] = color[1]
    rect[:, :, 2] = color[2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(rect, alpha, roi, 1.0 - alpha, 0)


def get_cyrillic_font(size=24):
    """Возвращает PIL-шрифт с поддержкой кириллицы."""
    # Список путей для поиска шрифтов (Linux, Windows, macOS)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux Debian/Ubuntu
        "/usr/share/fonts/TTF/DejaVuSans.ttf",                   # Linux Arch
        "C:\\Windows\\Fonts\\arial.ttf",                         # Windows
        "/Library/Fonts/Arial.ttf",                              # macOS
        "arial.ttf",                                             # Текущая директория
        "DejaVuSans.ttf",                                        # Локальная копия
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    # Если ничего не найдено — используем встроенный шрифт PIL (базовая поддержка Unicode)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def put_text_shadow(img, text, pos, font=cv2.FONT_HERSHEY_DUPLEX,
                    scale=0.75, color=C["white"], thickness=2):
    """Текст с тенью для читаемости на любом фоне."""
    x, y = pos
    # Для кириллицы используем PIL с шрифтом TrueType
    font_pil = get_cyrillic_font(int(scale * 32))
    if font_pil is not None:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            # Тень
            draw.text((x + 2, y + 2), text, font=font_pil, fill=(0, 0, 0))
            # Основной текст
            draw.text((x, y), text, font=font_pil, fill=(int(color[2]), int(color[1]), int(color[0])))
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    # Fallback на стандартный cv2.putText если PIL не работает
    cv2.putText(img, text, (x + 2, y + 2), font, scale, C["black"],
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color,
                thickness, cv2.LINE_AA)


def put_text_cv2(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                 scale=0.48, color=C["white"], thickness=1):
    """Рендеринг текста с кириллицей через PIL."""
    x, y = pos
    font_pil = get_cyrillic_font(int(scale * 32))
    if font_pil is not None:
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), text, font=font_pil, fill=(int(color[2]), int(color[1]), int(color[0])))
            img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_yolo_boxes(frame: np.ndarray, results, person_box: Optional[Tuple]):
    """
    Рисуем bounding boxes от YOLO.
    Синий = оператор, оранжевый = оборудование/объект.
    """
    if results is None:
        return

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            if cls == COCO_PERSON:
                color = C["blue"]
                label = f"Оператор {conf:.0%}"
            elif cls in COCO_EQUIPMENT:
                color = C["orange"]
                label = f"{COCO_EQUIPMENT[cls]} {conf:.0%}"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 9, y1),
                          color, -1)
            put_text_cv2(frame, label, (x1 + 3, y1 - 5),
                        scale=0.48, color=C["black"], thickness=1)


def draw_ppe_panel(frame: np.ndarray, st: AppState):
    """
    Панель СИЗ — левый верхний угол.
    Показывает прогресс-бары для каски и жилета.
    Исчезает после допуска.
    """
    if st.current not in (State.PPE_CHECK, State.ACCESS_GRANTED):
        return

    panel_items = [
        ("КАСКА",  st.helmet_cnt, st.helmet_ok),
        ("ЖИЛЕТ",  st.vest_cnt,   st.vest_ok),
    ]

    # Фон панели
    alpha_overlay(frame, 8, 8, 220, 8 + len(panel_items) * 50 + 12,
                  C["dark_bg"], alpha=0.80)
    cv2.rectangle(frame, (8, 8), (220, 8 + len(panel_items) * 50 + 12),
                  C["mid_gray"], 1)

    for i, (label, cnt, ok) in enumerate(panel_items):
        base_y = 32 + i * 50
        color  = C["green"] if ok else C["yellow"]
        mark   = "[OK]" if ok else "[--]"

        put_text_shadow(frame, f"{mark} {label}", (18, base_y),
                        scale=0.62, color=color)

        # Прогресс-бар
        bar_x1, bar_y1 = 18, base_y + 8
        bar_x2 = 208
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + 8),
                      (55, 55, 55), -1)
        fill = int((bar_x2 - bar_x1) * min(1.0, cnt / CONFIRM_FRAMES))
        if fill > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1),
                          (bar_x1 + fill, bar_y1 + 8), color, -1)


def draw_status_bar(frame: np.ndarray, st: AppState):
    """
    Нижняя строка состояния — постоянно видна.
    Содержит: заголовок состояния, подсказку, Safety Score + бар.

    Это главный "голос" системы — всегда видно, что делать дальше.
    Никакой бумажной инструкции: всё перед глазами в реальном времени.
    """
    h, w    = frame.shape[:2]
    BAR_H   = 88
    meta    = STATE_META[st.current]
    color   = meta["color"]

    # Фон
    alpha_overlay(frame, 0, h - BAR_H, w, h, C["dark_bg"], alpha=0.88)
    # Цветная полоска-акцент
    cv2.rectangle(frame, (0, h - BAR_H), (w, h - BAR_H + 4), color, -1)

    # Название состояния
    put_text_shadow(frame, meta["title"], (14, h - BAR_H + 30),
                    font=cv2.FONT_HERSHEY_DUPLEX, scale=0.70, color=color)

    # Подсказка (маленьким шрифтом)
    put_text_cv2(frame, meta["hint"], (14, h - BAR_H + 58),
                scale=0.48, color=C["white"])

    # ── Safety Score (правая часть) ───────────────────────────
    sx = w - 165
    put_text_cv2(frame, "SAFETY SCORE", (sx, h - BAR_H + 22),
                scale=0.40, color=(180, 180, 180))
    put_text_shadow(frame, f"{st.score:3d} / 100", (sx, h - BAR_H + 54),
                    font=cv2.FONT_HERSHEY_DUPLEX, scale=0.88, color=color)
    # Прогресс-бар очков
    bx1, bx2 = sx, sx + 152
    by = h - 18
    cv2.rectangle(frame, (bx1, by), (bx2, by + 9), (55, 55, 55), -1)
    fill_w = int((bx2 - bx1) * st.score / 100)
    if fill_w > 0:
        cv2.rectangle(frame, (bx1, by), (bx1 + fill_w, by + 9), color, -1)

    # Лого
    put_text_cv2(frame, "KASBOT v1.0", (14, h - 6),
                scale=0.38, color=(80, 80, 80))


def draw_ar_guide(frame: np.ndarray, st: AppState):
    """
    AR-стрелка и подсветка — показывает КУДА смотреть или ЧТО делать.

    Ключевое отличие от бумажной инструкции: пользователь не читает
    текст, а видит стрелку прямо в камере — как в навигаторе.
    Это снижает когнитивную нагрузку и убирает страх "не понять".
    """
    if st.current not in (State.STEP1_WORKSTATION, State.STEP2_POWER):
        return

    h, w = frame.shape[:2]
    st.phase += 0.10
    pulse = int(140 + 80 * np.sin(st.phase))  # 60..220 — плавная пульсация

    cx, cy  = w // 2, h // 2
    meta    = STATE_META[st.current]
    color   = meta["color"]

    if st.current == State.STEP1_WORKSTATION:
        # Стрелка ↓ — направляем взгляд вниз (на рабочую поверхность)
        pts = np.array([
            [cx,      cy + 70],
            [cx - 35, cy + 10],
            [cx - 15, cy + 10],
            [cx - 15, cy - 30],
            [cx + 15, cy - 30],
            [cx + 15, cy + 10],
            [cx + 35, cy + 10],
        ], np.int32)
        instruction = "Наведи КАМЕРУ ВНИЗ на рабочее место"
        # Прогресс поиска
        progress = min(1.0, st.target_cnt / CONFIRM_FRAMES)
    else:
        # Стрелка → — к кнопке справа
        pts = np.array([
            [cx + 70, cy],
            [cx + 10, cy - 35],
            [cx + 10, cy - 15],
            [cx - 30, cy - 15],
            [cx - 30, cy + 15],
            [cx + 10, cy + 15],
            [cx + 10, cy + 35],
        ], np.int32)
        instruction = "ПОДНИМИ РУКУ к кнопке питания"
        progress = min(1.0, st.hand_cnt / CONFIRM_FRAMES)

    # Рисуем стрелку с прозрачностью
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, pulse / 255.0, frame, 1 - pulse / 255.0, 0, frame)
    cv2.polylines(frame, [pts], True, C["white"], 1)

    # Текст инструкции по центру над стрелкой
    tw = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
    put_text_shadow(frame, instruction, ((w - tw) // 2, cy - 52),
                    font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.62, color=color)

    # Прогресс-бар под стрелкой (сколько ещё держать кадр)
    if 0 < progress < 1.0:
        bar_len = 180
        bx1 = (w - bar_len) // 2
        bx2 = bx1 + bar_len
        by  = cy + 100
        cv2.rectangle(frame, (bx1, by), (bx2, by + 10), (55, 55, 55), -1)
        cv2.rectangle(frame, (bx1, by),
                      (bx1 + int(bar_len * progress), by + 10), color, -1)
        put_text_cv2(frame, "Удержи кадр...", (bx1, by + 26),
                    scale=0.45, color=C["white"])


def draw_access_banner(frame: np.ndarray, st: AppState):
    """
    Баннер "ДОПУСК ВЫДАН" — ключевой момент системы.

    Психологически важный экран: новичок видит, что система
    его ЗАМЕТИЛА и ОДОБРИЛА. Это снижает тревогу первого дня
    и создаёт ощущение поддержки, которое важнее любой инструкции.
    """
    if st.current != State.ACCESS_GRANTED:
        return
    t = st.time_in_state()
    if t > 3.0:
        return
    # Плавное появление → затухание
    alpha = 0.75 * (1.0 - t / 3.0) if t > 0.5 else 0.75 * (t / 0.5)

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Зелёная рамка по краям кадра
    for th in [22, 14, 6]:
        cv2.rectangle(overlay, (th, th), (w - th, h - th - 88), C["green"], 3)

    # Затемнение центра
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    texts = [
        ("ДОПУСК ВЫДАН",     1.55, C["green"],  h // 2 - 25),
        ("ВХОД РАЗРЕШЁН",    0.85, C["white"],  h // 2 + 38),
        (f"+40 Safety Points!", 0.62, C["yellow"], h // 2 + 74),
    ]
    for text, scale, color, y in texts:
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)[0][0]
        put_text_shadow(frame, text, ((w - tw) // 2, y),
                        font=cv2.FONT_HERSHEY_DUPLEX, scale=scale, color=color,
                        thickness=2)


def draw_complete_screen(frame: np.ndarray, st: AppState):
    """
    Финальный экран — "как прохождение уровня в игре".

    Молодой сотрудник чувствует себя победителем первого дня.
    Safety Score фиксируется для отчёта наставника.
    Это самый важный экран для мотивации оставаться на заводе.
    """
    if st.current != State.COMPLETE:
        return

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    score_color = C["green"] if st.score >= 80 else C["yellow"]
    lines = [
        ("ПЕРВЫЙ ДЕНЬ",             1.6,  C["yellow"],  h // 2 - 120),
        ("УСПЕШНО ПРОЙДЕН!",        2.0,  C["green"],   h // 2 - 45),
        (f"Safety Score:  {st.score} / 100", 1.0, score_color, h // 2 + 50),
        ("Ты официально допущен к работе",   0.65, C["white"],  h // 2 + 100),
        ("━" * 42,                  0.45, C["mid_gray"], h // 2 + 130),
        ("[ Q ] выйти    [ R ] начать заново", 0.55, C["white"], h // 2 + 155),
    ]
    for text, scale, color, y in lines:
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)[0][0]
        put_text_shadow(frame, text, ((w - tw) // 2, y),
                        font=cv2.FONT_HERSHEY_DUPLEX, scale=scale, color=color)

    # Мигающая рамка (пульсирует)
    st.phase += 0.08
    edge_alpha = int(60 + 40 * np.sin(st.phase))
    ov2 = frame.copy()
    cv2.rectangle(ov2, (12, 12), (w - 12, h - 12), C["green"], 4)
    cv2.addWeighted(ov2, edge_alpha / 100.0, frame, 1 - edge_alpha / 100.0, 0, frame)


def draw_demo_hint(frame: np.ndarray, demo_mode: bool, st: AppState):
    """Подсказки по клавишам в демо-режиме (правый верхний угол)."""
    if not demo_mode:
        return
    h, w = frame.shape[:2]
    hints = [
        ("H  — каска",    st.sim_helmet),
        ("V  — жилет",    st.sim_vest),
        ("SPC — след.шаг", False),
        ("R  — сброс",    False),
    ]
    alpha_overlay(frame, w - 175, 6, w - 4, 6 + len(hints) * 22 + 8,
                  C["dark_bg"], alpha=0.72)
    for i, (text, active) in enumerate(hints):
        color = C["green"] if active else (130, 130, 130)
        put_text_cv2(frame, text, (w - 168, 26 + i * 22),
                    scale=0.43, color=color)


def draw_hand_status(frame: np.ndarray, hand_up: bool):
    """Иконка руки — показываем только на шаге включения питания."""
    if not MEDIAPIPE_AVAILABLE:
        return
    h, w = frame.shape[:2]
    color = C["green"] if hand_up else C["mid_gray"]
    text  = "РУКА ПОДНЯТА" if hand_up else "Рука не обнаружена"
    put_text_shadow(frame, text, (w // 2 - 90, 32),
                    font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.58, color=color,
                    thickness=1)


# ══════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="КАСБОТ — ИИ-наставник первого дня на заводе",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--demo",   action="store_true",
                        help="Демо-режим: H/V/SPACE управляют детекцией")
    parser.add_argument("--cam",    type=int, default=0,
                        help="Индекс камеры (default: 0)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # ── Открываем камеру ──────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"\n[ОШИБКА] Камера {args.cam} не открылась.")
        print("Попробуйте другой индекс: python main.py --cam 1")
        print("Или в демо-режиме без камеры: python main.py --demo\n")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Камера {args.cam}: {actual_w}×{actual_h}")

    # ── Загружаем YOLOv8n ─────────────────────────────────────────
    yolo = None
    if YOLO_AVAILABLE:
        try:
            print("[...] Загрузка YOLOv8n (~6MB, только первый раз)...")
            yolo = YOLO("yolov8n.pt")  # автозагрузка с ultralytics CDN
            print("[OK] YOLOv8n готов")
        except Exception as e:
            print(f"[WARN] YOLO недоступен: {e}")
            print("       Продолжаем с HSV-детекцией")

    # ── Состояние приложения ──────────────────────────────────────
    st = AppState()

    print("\n" + "─" * 56)
    print("  КАСБОТ запущен!")
    if args.demo:
        print("  ДЕМО-РЕЖИМ:  H=каска  V=жилет  SPACE=след.шаг")
    print("  Q — выход    R — рестарт")
    print("─" * 56 + "\n")

    yolo_results = None
    frame_idx    = 0
    fps_timer    = time.time()
    fps_val      = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Камера временно недоступна — ждём и повторяем
            print("[WARN] Потеря кадра, продолжаю...")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)   # зеркало — "selfie"-режим
        frame_idx += 1
        fh, fw = frame.shape[:2]

        # ── FPS (для отладки) ─────────────────────────────────────
        if frame_idx % 30 == 0:
            fps_val   = 30.0 / (time.time() - fps_timer + 1e-9)
            fps_timer = time.time()

        # ══ CV-СЛОЙ: ДЕТЕКЦИЯ ════════════════════════════════════
        # YOLO запускается не каждый кадр (экономия CPU)
        if yolo and frame_idx % YOLO_INTERVAL == 0:
            try:
                yolo_results = yolo.predict(frame, verbose=False, conf=0.38,
                                            classes=list({COCO_PERSON} |
                                                         set(COCO_EQUIPMENT)))
            except Exception as e:
                yolo_results = None

        person_box = get_largest_person(yolo_results)

        try:
            ppe = detect_ppe_colors(frame, person_box)
        except Exception:
            ppe = {"helmet": False, "vest": False}

        try:
            hand_up = (detect_raised_hand(frame)
                       if st.current == State.STEP2_POWER else False)
        except Exception:
            hand_up = False

        try:
            target_found = detect_equipment(yolo_results, fh, fw)
        except Exception:
            target_found = False

        # ══ FSM-СЛОЙ: ОБНОВЛЕНИЕ ═════════════════════════════════
        update_fsm(st, ppe, hand_up, target_found)

        # ══ UI-СЛОЙ: РЕНДЕРИНГ ═══════════════════════════════════
        draw_yolo_boxes(frame, yolo_results, person_box)
        draw_hand_landmarks(frame)

        draw_ppe_panel(frame, st)
        draw_ar_guide(frame, st)
        draw_access_banner(frame, st)
        draw_complete_screen(frame, st)
        draw_status_bar(frame, st)

        if st.current == State.STEP2_POWER:
            draw_hand_status(frame, hand_up)

        draw_demo_hint(frame, args.demo, st)

        # FPS в углу
        put_text_cv2(frame, f"{fps_val:.0f} fps",
                    (fw - 72, fh - 96),
                    scale=0.40, color=(80, 80, 80))

        cv2.imshow("КАСБОТ — ИИ-наставник | Technostrelka 2026", frame)

        # ══ КЛАВИАТУРА ═══════════════════════════════════════════
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord('r'):
            st           = AppState()
            yolo_results = None
            frame_idx    = 0
            print("[INFO] Рестарт!")

        elif args.demo:
            if key == ord('h'):
                st.sim_helmet = not st.sim_helmet
                print(f"[DEMO] Каска: {'ON' if st.sim_helmet else 'OFF'}")
            elif key == ord('v'):
                st.sim_vest = not st.sim_vest
                print(f"[DEMO] Жилет: {'ON' if st.sim_vest else 'OFF'}")
            elif key == 32:  # SPACE — принудительный переход
                if st.current in NEXT_STATE:
                    target = NEXT_STATE[st.current]
                    old_min = st.__class__.__dataclass_fields__  # noqa
                    # временно снижаем порог времени
                    orig = MIN_STATE_TIME
                    import builtins
                    # Переходим напрямую
                    st.score = min(100, st.score + STATE_META[target]["points"])
                    st.current = target
                    st.state_since = time.time() - MIN_STATE_TIME  # пропустить задержку
                    st.hand_cnt = 0
                    st.target_cnt = 0
                    print(f"[DEMO] → {st.current.name}")

    # ── Завершение ────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    mp_h = get_mp_hands()
    if mp_h:
        mp_h.close()
    print(f"\n[КАСБОТ] Сессия завершена. Итоговый Safety Score: {st.score}/100")


if __name__ == "__main__":
    main()
