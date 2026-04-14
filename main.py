import time

import cv2
import numpy as np

from camera import CameraManager
from game_logic import PoseBattleGame, load_pose_references
from pose_compare import extract_joint_angles, performance_message, score_pose
from pose_detector import PoseDetector
from ui import build_target_panel, draw_game_hud, draw_start_screen, draw_winner_overlay


WINDOW_NAME = "Pose Battle Game"
TOTAL_ROUNDS = 5
COUNTDOWN_SECONDS = 5.0
RESULT_SHOW_SECONDS = 2.5


def main():
    poses = load_pose_references("poses")
    game = PoseBattleGame(poses, poses_per_round=TOTAL_ROUNDS)

    camera = CameraManager()
    detector = PoseDetector()

    state = "start"  # start | countdown | result | game_over
    state_start_time = time.time() 
    msg_p1 = ""
    msg_p2 = ""

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Failed to read camera frame.")
                break

            frame = cv2.flip(frame, 1)
            _, w = frame.shape[:2]
            mid = w // 2

            left = frame[:, :mid]
            right = frame[:, mid:]

            lm_p1, left_vis = detector.detect(left)
            lm_p2, right_vis = detector.detect(right)
            game_view = cv2.hconcat([left_vis, right_vis])

            current_pose = game.get_current_pose()
            pose_name = current_pose.display_name if current_pose else "None"
            panel_w = max(200, int(game_view.shape[1] * 0.25))
            target_panel = build_target_panel(
                current_pose.image if current_pose else None,
                pose_name,
                panel_width=panel_w,
                panel_height=game_view.shape[0],
            )

            countdown_text = ""
            if state == "start":
                draw_start_screen(game_view)

            elif state == "countdown":
                elapsed = time.time() - state_start_time
                remaining = COUNTDOWN_SECONDS - elapsed
                if remaining > 0:
                    countdown_text = str(int(np.ceil(remaining)))
                else:
                    countdown_text = "POSE"

                    p1_angles = extract_joint_angles(lm_p1)
                    p2_angles = extract_joint_angles(lm_p2)
                    target_angles = current_pose.target_angles if current_pose else {}

                    p1_score = score_pose(p1_angles, target_angles)
                    p2_score = score_pose(p2_angles, target_angles)

                    msg_p1 = performance_message(p1_score)
                    msg_p2 = performance_message(p2_score)
                    game.register_round_scores(p1_score, p2_score)

                    state = "game_over" if game.is_finished() else "result"
                    state_start_time = time.time()

            elif state == "result":
                elapsed = time.time() - state_start_time
                if elapsed >= RESULT_SHOW_SECONDS:
                    state = "countdown"
                    state_start_time = time.time()

            elif state == "game_over":
                draw_winner_overlay(game_view, game.winner_text())

            draw_game_hud(
                game_view,
                game.round_label(),
                game.player_totals["p1"],
                game.player_totals["p2"],
                game.last_round_scores["p1"],
                game.last_round_scores["p2"],
                countdown_text=countdown_text,
                msg_p1=msg_p1 if state in {"result", "game_over"} else "",
                msg_p2=msg_p2 if state in {"result", "game_over"} else "",
            )

            full_view = cv2.hconcat([game_view, target_panel])
            cv2.imshow(WINDOW_NAME, full_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if state == "start" and key == ord(" "):
                game.reset()
                state = "countdown"
                state_start_time = time.time()
                msg_p1, msg_p2 = "", ""

            if state == "game_over" and key == ord("r"):
                game.reset()
                state = "countdown"
                state_start_time = time.time()
                msg_p1, msg_p2 = "", ""

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
