import os
import pickle

import numpy as np
import xgboost as xgb

from pacman.data_core import EvenType, event_append

# Maps label string → pygame EvenType that Pacman's event_handler understands
_ACTION_TO_EVENT = {
    "up":    EvenType.UP_BTN,
    "down":  EvenType.DONW_BTN,
    "left":  EvenType.LEFT_BTN,
    "right": EvenType.RIGHT_BTN,
}

_MODEL_SUBDIR = "model"
_MODEL_FILENAME = "pacman_model.json"
_LE_FILENAME = "label_encoder.pkl"

_instance = None


def AutoPlayer() -> "AutoPlayAgent":
    global _instance
    if _instance is None:
        _instance = AutoPlayAgent()
    return _instance


class AutoPlayAgent:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._model: xgb.XGBClassifier | None = None
        self._label_classes: list | None = None   # e.g. ['down','left','right','up']
        self._action_history: list = []
        self._HISTORY_LEN = 6
        self._stuck_cooldown = 0
        self._last_action: str | None = None

    def enable(self, model_dir: str = "assignment") -> None:
        model_path = os.path.join(model_dir, _MODEL_SUBDIR, _MODEL_FILENAME)
        le_path = os.path.join(model_dir, _MODEL_SUBDIR, _LE_FILENAME)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[AutoPlayer] Model not found at {model_path}\n"
                "Train first with:  python -m assignment.train"
            )
        if not os.path.exists(le_path):
            raise FileNotFoundError(
                f"[AutoPlayer] Label encoder not found at {le_path}\n"
                "Train first with:  python -m assignment.train"
            )

        self._model = xgb.XGBClassifier()
        self._model.load_model(model_path)

        with open(le_path, "rb") as f:
            le = pickle.load(f)
        self._label_classes = list(le.classes_)

        self._enabled = True
        print(f"[AutoPlayer] Loaded model from {model_path}")
        print(f"[AutoPlayer] Actions: {self._label_classes}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def step(self, scene) -> None:
        if not self._enabled:
            return

        features = self._extract_features(scene)
        action = self._predict(features)
        action = self._suppress_oscillation(action)
        self._post_event(action)

    @staticmethod
    def _get_cell(rect):
        from pacman.misc.cell_util import CellUtil
        return CellUtil.get_cell(rect)

    def _extract_features(self, scene) -> np.ndarray:
        from math import hypot
        from pacman.data_core.enums import FruitStateEnum, GhostStateEnum

        pacman = scene.pacman
        ghosts = [scene.blinky, scene.pinky, scene.inky, scene.clyde]
        seeds_obj = scene._MainScene__seeds
        fruit_obj = scene._MainScene__fruit
        hp_obj = scene.hp

        TOTAL_SEEDS = 242
        DANGEROUS = (GhostStateEnum.CHASE, GhostStateEnum.SCATTER)

        px, py = self._get_cell(pacman.rect)

        # Direction one-hot  (rotate: 0=right 1=down 2=left 3=up)
        rotate = pacman.rotate
        moving = pacman.speed > 0
        dir_right = int(moving and rotate == 0)
        dir_left = int(moving and rotate == 2)
        dir_up = int(moving and rotate == 3)
        dir_down = int(moving and rotate == 1)

        # Wall flags  (movement_cell: [right, down, left, up], True=passable)
        walls = pacman.movement_cell(self._get_cell(pacman.rect))
        wall_right = int(not walls[0])
        wall_left = int(not walls[2])
        wall_up = int(not walls[3])
        wall_down = int(not walls[1])

        lives = int(hp_obj)

        # Ghost features
        ghost_vals = []
        for ghost in ghosts:
            gx, gy = self._get_cell(ghost.rect)
            ghost_vals += [gx, gy,
                           round(hypot(gx - px, gy - py), 4),
                           int(ghost.state in DANGEROUS)]

        # Pellet features
        seeds_grid = seeds_obj._SeedContainer__seeds
        seeds_remaining = seeds_obj._SeedContainer__seeds_counts
        seeds_eaten_cnt = TOTAL_SEEDS - seeds_remaining

        ndx, ndy, ndist = self._nearest_seed(px, py, seeds_grid)
        sr, sl, sd, su = self._directional_seeds(px, py, seeds_grid)

        # Energizer features
        from pacman.misc.cell_util import CellUtil
        energizers = seeds_obj._SeedContainer__energizers
        en_cells = [CellUtil.get_cell(e.rect) for e in energizers]
        en_remaining = len(en_cells)
        if en_cells:
            candidates = sorted(
                [(ex - px, ey - py, hypot(ex - px, ey - py)) for ex, ey in en_cells],
                key=lambda t: t[2],
            )
            en_dx, en_dy, en_dist = candidates[0]
        else:
            en_dx, en_dy, en_dist = 0, 0, 0.0

        fruit_active = int(fruit_obj.state == FruitStateEnum.ACTIVE)
        seeds_ratio = round(seeds_eaten_cnt / TOTAL_SEEDS, 6)

        row = [
            px, py,
            dir_right, dir_left, dir_up, dir_down,
            wall_right, wall_left, wall_up, wall_down,
            lives,
            *ghost_vals,
            ndx, ndy, round(ndist, 4),
            sr, sl, sd, su,
            en_dx, en_dy, round(en_dist, 4), en_remaining,
            fruit_active,
            seeds_ratio,
        ]
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def _predict(self, features: np.ndarray) -> str:
        idx = int(self._model.predict(features)[0])
        return self._label_classes[idx]

    _OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}

    def _suppress_oscillation(self, action: str) -> str:
        if self._stuck_cooldown > 0:
            self._stuck_cooldown -= 1
            return self._last_action

        self._action_history.append(action)
        if len(self._action_history) > self._HISTORY_LEN:
            self._action_history.pop(0)

        if len(self._action_history) == self._HISTORY_LEN:
            unique = set(self._action_history)
            if (len(unique) == 2 and
                    self._OPPOSITE.get(list(unique)[0]) == list(unique)[1]):
                self._stuck_cooldown = 10
                self._action_history.clear()

        self._last_action = action
        return action

    @staticmethod
    def _post_event(action: str) -> None:
        event_type = _ACTION_TO_EVENT.get(action)
        if event_type is not None:
            event_append(event_type)

    @staticmethod
    def _nearest_seed(px: int, py: int, seeds_grid: list) -> tuple:
        from math import hypot
        best_dx, best_dy, best_dist = 0, 0, float("inf")
        for gy, row in enumerate(seeds_grid):
            for gx, present in enumerate(row):
                if present:
                    d = hypot(gx - px, gy - py)
                    if d < best_dist:
                        best_dist = d
                        best_dx, best_dy = gx - px, gy - py
        if best_dist == float("inf"):
            return 0, 0, 0.0
        return best_dx, best_dy, best_dist

    @staticmethod
    def _directional_seeds(px: int, py: int, seeds_grid: list) -> tuple:
        rows = len(seeds_grid)
        cols = len(seeds_grid[0]) if rows else 0

        def count(x_range, y_range) -> int:
            total = 0
            for gy in y_range:
                if not (0 <= gy < rows):
                    continue
                for gx in x_range:
                    if 0 <= gx < cols and seeds_grid[gy][gx]:
                        total += 1
            return total

        right = count(range(px,     px + 5), range(py - 2, py + 3))
        left = count(range(px - 5, px),     range(py - 2, py + 3))
        down = count(range(px - 2, px + 3), range(py,     py + 5))
        up = count(range(px - 2, px + 3), range(py - 5, py))
        return right, left, down, up
