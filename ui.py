"""
Pose Battle Game – Modern, responsive UI rendering.
All positions and sizes are proportional to frame dimensions.
"""

from typing import Optional
import math
import time

import cv2
import numpy as np

# ─── Colour Palette ──────────────────────────────────────────────────────────
# Colours are BGR for OpenCV
COL_BG_DARK      = (20, 20, 25)
COL_BG_PANEL     = (30, 30, 38)
COL_P1_ACCENT    = (255, 229, 0)      # Cyan / teal  #00E5FF
COL_P2_ACCENT    = (129, 64, 255)     # Magenta / pink #FF4081
COL_GOLD         = (64, 215, 255)     # Warm gold #FFD740
COL_WHITE        = (240, 240, 245)
COL_WHITE_DIM    = (180, 180, 190)
COL_GREEN_SCORE  = (120, 255, 180)
COL_RED_SCORE    = (100, 100, 255)
COL_OVERLAY      = (15, 15, 20)
COL_COUNTDOWN    = (0, 240, 255)      # Bright yellow-ish


# ─── Helper utilities ────────────────────────────────────────────────────────

def _alpha_rect(frame, pt1, pt2, colour, alpha):
    """Draw a filled rectangle with alpha blending."""
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1, pt2, colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _rounded_rect(frame, pt1, pt2, colour, alpha, radius=12):
    """Draw a rounded-corner filled rectangle with alpha."""
    overlay = frame.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    # Clamp radius
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(overlay, pt1, pt2, colour, -1)
    else:
        # Draw the main body rectangles
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), colour, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), colour, -1)
        # Four corner circles
        cv2.circle(overlay, (x1 + r, y1 + r), r, colour, -1)
        cv2.circle(overlay, (x2 - r, y1 + r), r, colour, -1)
        cv2.circle(overlay, (x1 + r, y2 - r), r, colour, -1)
        cv2.circle(overlay, (x2 - r, y2 - r), r, colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _vignette(frame, strength=0.45):
    """Apply a subtle vignette darkening around edges."""
    h, w = frame.shape[:2]
    X = cv2.getGaussianKernel(w, w * 0.55)
    Y = cv2.getGaussianKernel(h, h * 0.55)
    M = Y @ X.T
    M = M / M.max()
    M = (1.0 - strength) + strength * M
    for c in range(3):
        frame[:, :, c] = np.clip(frame[:, :, c] * M, 0, 255).astype(np.uint8)


def _put_text_shadow(frame, text, org, font, scale, colour, thickness, shadow_offset=2):
    """Draw text with a drop shadow for readability."""
    sx, sy = org[0] + shadow_offset, org[1] + shadow_offset
    cv2.putText(frame, text, (sx, sy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, colour, thickness, cv2.LINE_AA)


def _put_text_centered(frame, text, center_x, y, font, scale, colour, thickness):
    """Draw text centered horizontally at (center_x, y)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (center_x - tw // 2, y)
    _put_text_shadow(frame, text, org, font, scale, colour, thickness)


def _draw_gradient_bar(frame, pt1, pt2, col_top, col_bot, alpha=0.7):
    """Draw a vertical gradient rectangle with alpha blending."""
    x1, y1 = pt1
    x2, y2 = pt2
    bar_h = y2 - y1
    if bar_h <= 0:
        return
    overlay = frame.copy()
    for i in range(bar_h):
        t = i / max(1, bar_h - 1)
        b = int(col_top[0] * (1 - t) + col_bot[0] * t)
        g = int(col_top[1] * (1 - t) + col_bot[1] * t)
        r = int(col_top[2] * (1 - t) + col_bot[2] * t)
        cv2.line(overlay, (x1, y1 + i), (x2, y1 + i), (b, g, r), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_accent_line(frame, x, y, length, colour, thickness=3):
    """Draw a short horizontal accent line."""
    cv2.line(frame, (x, y), (x + length, y), colour, thickness, cv2.LINE_AA)


# ─── Start Screen ────────────────────────────────────────────────────────────

def draw_start_screen(frame: np.ndarray):
    """Modern start screen with vignette, glowing title, and pulsing instruction."""
    h, w = frame.shape[:2]

    # Darken background
    _alpha_rect(frame, (0, 0), (w, h), COL_OVERLAY, 0.6)
    _vignette(frame, 0.5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(0.8, min(2.5, w / 520))
    sub_scale = max(0.5, min(1.2, w / 1100))

    # Title glow effect (draw larger blurred text behind)
    (tw, _), _ = cv2.getTextSize("POSE BATTLE", font, title_scale, 3)
    tx = w // 2 - tw // 2
    ty = int(h * 0.40)

    # Outer glow
    glow_col = (80, 200, 255)
    cv2.putText(frame, "POSE BATTLE", (tx, ty), font, title_scale, glow_col, 7, cv2.LINE_AA)
    # Main title
    _put_text_shadow(frame, "POSE BATTLE", (tx, ty), font, title_scale, COL_GOLD, 3, shadow_offset=3)

    # Subtitle
    subtitle = "GAME"
    (sw, _), _ = cv2.getTextSize(subtitle, font, title_scale * 0.7, 2)
    sx = w // 2 - sw // 2
    sy = ty + int(title_scale * 45)
    _put_text_shadow(frame, subtitle, (sx, sy), font, title_scale * 0.7, COL_WHITE, 2)

    # Pulsing instruction
    pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
    alpha_val = int(160 + 95 * pulse)
    inst_col = (alpha_val, alpha_val, alpha_val)

    inst_text = "Press SPACE to Start"
    _put_text_centered(frame, inst_text, w // 2, int(h * 0.62), font, sub_scale, inst_col, 2)

    # Decorative accent lines
    line_y = int(h * 0.68)
    line_len = int(w * 0.15)
    _draw_accent_line(frame, w // 2 - line_len - 10, line_y, line_len, COL_P1_ACCENT, 2)
    _draw_accent_line(frame, w // 2 + 10, line_y, line_len, COL_P2_ACCENT, 2)

    # Tagline
    tag = "Strike a Pose. Win the Battle."
    _put_text_centered(frame, tag, w // 2, int(h * 0.75), font, sub_scale * 0.65, COL_WHITE_DIM, 1)


# ─── Target Panel ────────────────────────────────────────────────────────────

def build_target_panel(
    target_img: Optional[np.ndarray], pose_name: str, panel_width: int, panel_height: int
) -> np.ndarray:
    """Card-style target panel with rounded visual elements."""
    panel = np.full((panel_height, panel_width, 3), COL_BG_DARK[0], dtype=np.uint8)
    panel[:, :] = COL_BG_DARK

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(8, int(panel_width * 0.06))
    header_h = max(55, int(panel_height * 0.12))

    # Header background
    _rounded_rect(panel, (pad, pad), (panel_width - pad, pad + header_h), (40, 40, 50), 0.8, radius=10)

    # "Target Pose" label
    label_scale = max(0.45, min(0.8, panel_width / 400))
    _put_text_shadow(panel, "TARGET POSE", (pad + 12, pad + int(header_h * 0.42)),
                     font, label_scale * 0.75, COL_WHITE_DIM, 1, shadow_offset=1)

    # Pose name
    name_scale = max(0.45, min(0.85, panel_width / 350))
    _put_text_shadow(panel, pose_name.upper(), (pad + 12, pad + int(header_h * 0.82)),
                     font, name_scale, COL_GOLD, 2, shadow_offset=1)

    # Accent line under header
    _draw_accent_line(panel, pad + 10, pad + header_h + 4, panel_width - 2 * pad - 20, COL_P1_ACCENT, 2)

    # Image area
    img_top = pad + header_h + 16
    img_bottom = panel_height - pad
    img_h = img_bottom - img_top
    img_w = panel_width - 2 * pad

    if img_h <= 0 or img_w <= 0:
        return panel

    # Image card background
    _rounded_rect(panel, (pad, img_top), (pad + img_w, img_top + img_h), (35, 35, 45), 0.6, radius=8)

    if target_img is None:
        _put_text_centered(panel, "No image", panel_width // 2, img_top + img_h // 2,
                          font, label_scale * 0.8, COL_WHITE_DIM, 1)
        return panel

    # Fit and center the image
    resized = _fit_image(target_img, img_w - 16, img_h - 16)
    ry, rx = resized.shape[:2]
    y_off = img_top + (img_h - ry) // 2
    x_off = pad + (img_w - rx) // 2
    # Ensure bounds
    y_off = max(img_top, y_off)
    x_off = max(pad, x_off)
    y_end = min(y_off + ry, panel_height)
    x_end = min(x_off + rx, panel_width)
    ry_actual = y_end - y_off
    rx_actual = x_end - x_off
    if ry_actual > 0 and rx_actual > 0:
        panel[y_off:y_end, x_off:x_end] = resized[:ry_actual, :rx_actual]

    # Subtle border around image region
    cv2.rectangle(panel, (pad, img_top), (pad + img_w, img_top + img_h), (60, 60, 70), 1, cv2.LINE_AA)

    return panel


def _fit_image(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = min(max_w / max(1, w), max_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─── Game HUD ────────────────────────────────────────────────────────────────

def draw_game_hud(
    frame: np.ndarray,
    round_label: str,
    p1_total: float,
    p2_total: float,
    p1_last: float,
    p2_last: float,
    countdown_text: str = "",
    msg_p1: str = "",
    msg_p2: str = "",
):
    """Modern responsive HUD with gradient header bar and styled elements."""
    h, w = frame.shape[:2]
    mid_x = w // 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top HUD bar ──
    bar_h = max(50, int(h * 0.15))
    _draw_gradient_bar(frame, (0, 0), (w, bar_h), (35, 35, 45), (20, 20, 28), alpha=0.75)

    # Divider line (center, subtle)
    cv2.line(frame, (mid_x, 0), (mid_x, h), (60, 60, 70), 1, cv2.LINE_AA)

    # Responsive font sizes
    name_scale = max(0.5, min(1.1, w / 1000))
    score_scale = max(0.4, min(0.8, w / 1200))
    round_scale = max(0.4, min(0.7, w / 1300))

    # Player names with accent bars
    p1_x = int(w * 0.02)
    p2_x = mid_x + int(w * 0.02)
    name_y = int(bar_h * 0.35)

    # Player 1 accent bar + name
    _draw_accent_line(frame, p1_x, name_y + 5, int(w * 0.04), COL_P1_ACCENT, 3)
    _put_text_shadow(frame, "PLAYER 1", (p1_x + int(w * 0.05), name_y + 5),
                     font, name_scale, COL_P1_ACCENT, 2, shadow_offset=1)

    # Player 2 accent bar + name
    _draw_accent_line(frame, p2_x, name_y + 5, int(w * 0.04), COL_P2_ACCENT, 3)
    _put_text_shadow(frame, "PLAYER 2", (p2_x + int(w * 0.05), name_y + 5),
                     font, name_scale, COL_P2_ACCENT, 2, shadow_offset=1)

    # Scores
    score_y1 = int(bar_h * 0.60)
    score_y2 = int(bar_h * 0.82)

    # P1 scores
    _put_text_shadow(frame, f"Total: {p1_total:.1f}", (p1_x + 8, score_y1),
                     font, score_scale, COL_WHITE, 1, shadow_offset=1)
    _put_text_shadow(frame, f"Last:  {p1_last:.1f}", (p1_x + 8, score_y2),
                     font, score_scale * 0.85, COL_GREEN_SCORE, 1, shadow_offset=1)

    # P2 scores
    _put_text_shadow(frame, f"Total: {p2_total:.1f}", (p2_x + 8, score_y1),
                     font, score_scale, COL_WHITE, 1, shadow_offset=1)
    _put_text_shadow(frame, f"Last:  {p2_last:.1f}", (p2_x + 8, score_y2),
                     font, score_scale * 0.85, COL_GREEN_SCORE, 1, shadow_offset=1)

    # Round badge (centered pill)
    (rw, rh), _ = cv2.getTextSize(round_label, font, round_scale, 2)
    rx = mid_x - rw // 2
    ry = int(bar_h * 0.28)
    pill_pad_x = 14
    pill_pad_y = 6
    _rounded_rect(frame, (rx - pill_pad_x, ry - rh - pill_pad_y),
                  (rx + rw + pill_pad_x, ry + pill_pad_y), (50, 50, 65), 0.7, radius=10)
    _put_text_shadow(frame, round_label, (rx, ry), font, round_scale, COL_GOLD, 2, shadow_offset=1)

    # ── Countdown ──
    if countdown_text:
        cd_scale = max(1.5, min(4.0, w / 350))
        # Glow circle behind countdown
        glow_r = int(min(w, h) * 0.08)
        glow_center = (w // 2, h // 2)
        for i in range(3):
            r = glow_r + i * 8
            alpha = 0.12 - i * 0.03
            overlay = frame.copy()
            cv2.circle(overlay, glow_center, r, COL_COUNTDOWN, -1)
            cv2.addWeighted(overlay, max(0.02, alpha), frame, 1 - max(0.02, alpha), 0, frame)

        _put_text_centered(frame, countdown_text, w // 2, h // 2 + int(cd_scale * 8),
                          font, cd_scale, COL_COUNTDOWN, 4)

    # ── Performance messages ──
    if msg_p1:
        msg_col = COL_GREEN_SCORE if "Perfect" in msg_p1 else COL_GOLD if "Almost" in msg_p1 else COL_RED_SCORE
        msg_scale = max(0.45, min(0.8, w / 1200))
        msg_y = h - int(h * 0.05)
        # Background pill for message
        (mw, mh), _ = cv2.getTextSize(msg_p1, font, msg_scale, 2)
        _rounded_rect(frame, (p1_x, msg_y - mh - 8), (p1_x + mw + 20, msg_y + 8),
                      (25, 25, 35), 0.65, radius=8)
        _put_text_shadow(frame, msg_p1, (p1_x + 10, msg_y), font, msg_scale, msg_col, 2, shadow_offset=1)

    if msg_p2:
        msg_col = COL_GREEN_SCORE if "Perfect" in msg_p2 else COL_GOLD if "Almost" in msg_p2 else COL_RED_SCORE
        msg_scale = max(0.45, min(0.8, w / 1200))
        msg_y = h - int(h * 0.05)
        (mw, mh), _ = cv2.getTextSize(msg_p2, font, msg_scale, 2)
        _rounded_rect(frame, (p2_x, msg_y - mh - 8), (p2_x + mw + 20, msg_y + 8),
                      (25, 25, 35), 0.65, radius=8)
        _put_text_shadow(frame, msg_p2, (p2_x + 10, msg_y), font, msg_scale, msg_col, 2, shadow_offset=1)


# ─── Winner Overlay ──────────────────────────────────────────────────────────

def draw_winner_overlay(frame: np.ndarray, winner_text: str):
    """Dramatic game-over overlay with glow and styled winner text."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dark overlay with vignette
    _alpha_rect(frame, (0, 0), (w, h), COL_OVERLAY, 0.55)
    _vignette(frame, 0.4)

    title_scale = max(0.9, min(2.5, w / 500))
    sub_scale = max(0.5, min(1.2, w / 900))
    inst_scale = max(0.4, min(0.85, w / 1200))

    # "GAME OVER" with glow
    go_text = "GAME OVER"
    (gw, gh), _ = cv2.getTextSize(go_text, font, title_scale, 3)
    gx = w // 2 - gw // 2
    gy = int(h * 0.38)

    # Outer glow
    cv2.putText(frame, go_text, (gx, gy), font, title_scale, (50, 50, 200), 7, cv2.LINE_AA)
    _put_text_shadow(frame, go_text, (gx, gy), font, title_scale, COL_GOLD, 3, shadow_offset=3)

    # Winner text with colour matching
    if "Player 1" in winner_text:
        win_col = COL_P1_ACCENT
    elif "Player 2" in winner_text:
        win_col = COL_P2_ACCENT
    elif "Tie" in winner_text:
        win_col = COL_GOLD
    else:
        win_col = COL_WHITE

    # Winner pill background
    (ww, wh), _ = cv2.getTextSize(winner_text, font, sub_scale, 2)
    wx = w // 2 - ww // 2
    wy = int(h * 0.52)
    _rounded_rect(frame, (wx - 20, wy - wh - 12), (wx + ww + 20, wy + 12),
                  (30, 30, 40), 0.7, radius=14)
    _put_text_shadow(frame, winner_text, (wx, wy), font, sub_scale, win_col, 2, shadow_offset=2)

    # Decorative accent lines
    line_y = int(h * 0.58)
    line_len = int(w * 0.12)
    _draw_accent_line(frame, w // 2 - line_len - 10, line_y, line_len, COL_P1_ACCENT, 2)
    _draw_accent_line(frame, w // 2 + 10, line_y, line_len, COL_P2_ACCENT, 2)

    # Instructions
    inst_text = "Press R to Play Again  |  Q to Quit"
    pulse = 0.5 + 0.5 * math.sin(time.time() * 2.5)
    alpha_val = int(150 + 100 * pulse)
    inst_col = (alpha_val, alpha_val, alpha_val)
    _put_text_centered(frame, inst_text, w // 2, int(h * 0.68), font, inst_scale, inst_col, 2)
