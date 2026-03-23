import csv
import os
from datetime import datetime
from math import hypot

from pacman.data_core.enums import FruitStateEnum, GhostStateEnum

# feature list
FEATURE_COLUMNS = [
    # episode
    "episode_id",
    # self
    "pacman_x", "pacman_y",
    "moving_dir",
    "moving_axis",
    "is_reversing_right", "is_reversing_left", "is_reversing_up", "is_reversing_down",
    "wall_right", "wall_left", "wall_up", "wall_down",
    "lives_remaining",
    # gosts
    "ghost_b_x", "ghost_b_y", "ghost_b_dist", "ghost_b_is_dangerous",
    "ghost_p_x", "ghost_p_y", "ghost_p_dist", "ghost_p_is_dangerous",
    "ghost_i_x", "ghost_i_y", "ghost_i_dist", "ghost_i_is_dangerous",
    "ghost_c_x", "ghost_c_y", "ghost_c_dist", "ghost_c_is_dangerous",
    # seeds
    "nearest_seed_dx", "nearest_seed_dy", "nearest_seed_dist",
    "seeds_right", "seeds_left", "seeds_down", "seeds_up",
    # energizers
    "nearest_energizer_dx", "nearest_energizer_dy", "nearest_energizer_dist",
    "energizers_remaining",
    # others
    "fruit_active",
    "seeds_eaten_ratio",
    # label
    "action",
]

ACTION_NONE = "none"
ACTION_UP = "up"
ACTION_DOWN = "down"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"

TOTAL_SEEDS = 242

_instance = None


def Recorder() -> "DataRecorder":
    global _instance
    if _instance is None:
        _instance = DataRecorder()
    return _instance


class DataRecorder:
    def __init__(self) -> None:
        self._enabled: bool = False
        self._file = None
        self._writer = None
        self._current_action: str = ACTION_NONE
        self._episode_id: int = 0

    def next_episode(self) -> None:
        if not self._enabled:
            return
        self._episode_id += 1
        self._current_action = ACTION_NONE
        print(f"[Recorder] Episode {self._episode_id} started.")

    def enable(self, output_dir: str = "recordings") -> None:
        self._enabled = True
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"game_{timestamp}.csv")
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FEATURE_COLUMNS)
        self._writer.writeheader()
        print(f"[Recorder] Recording enabled -> {path}")

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
            print("[Recorder] Recording saved.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_action(self, action: str) -> None:
        self._current_action = action

    def record_frame(self, scene) -> None:
        if not self._enabled or self._writer is None:
            return
        row = self._extract_features(scene)
        pacman = scene.pacman
        _rotate_to_action = {0: ACTION_RIGHT, 1: ACTION_DOWN, 2: ACTION_LEFT, 3: ACTION_UP}
        if pacman.speed > 0 and pacman.rotate in _rotate_to_action:
            action = _rotate_to_action[pacman.rotate]
        else:
            action = ACTION_NONE
        row["action"] = action
        row["episode_id"] = self._episode_id
        self._writer.writerow(row)

    @staticmethod
    def _get_cell(rect):
        from pacman.misc.cell_util import CellUtil
        return CellUtil.get_cell(rect)

    def _extract_features(self, scene) -> dict:
        pacman = scene.pacman
        ghosts = [scene.blinky, scene.pinky, scene.inky, scene.clyde]
        seeds_obj = scene._MainScene__seeds
        fruit_obj = scene._MainScene__fruit
        hp_obj = scene.hp
        seeds_eaten = scene._MainScene__seeds_eaten
        px, py = self._get_cell(pacman.rect)
        # Direction one-hot
        # rotate: 0=right, 1=down, 2=left, 3=up  (None when stopped)
        rotate = pacman.rotate
        moving = pacman.speed > 0
        if not moving:
            moving_dir = 0
            moving_axis = 0
        elif rotate == 0:
            moving_dir = 1
            moving_axis = 1
        elif rotate == 2:
            moving_dir = 2
            moving_axis = 1
        elif rotate == 3:
            moving_dir = 3
            moving_axis = 2
        else:
            moving_dir = 4
            moving_axis = 2
        is_reversing_right = int(moving and rotate == 2)
        is_reversing_left = int(moving and rotate == 0)
        is_reversing_up = int(moving and rotate == 1)
        is_reversing_down = int(moving and rotate == 3)
        # Wall flags (movement_cell returns [right, down, left, up])
        # True = passable; invert to get wall=1
        walls = pacman.movement_cell(self._get_cell(pacman.rect))
        wall_right = int(not walls[0])
        wall_left = int(not walls[2])
        wall_up = int(not walls[3])
        wall_down = int(not walls[1])
        lives = int(hp_obj)
        DANGEROUS = (GhostStateEnum.CHASE, GhostStateEnum.SCATTER)
        ghost_feats = {}
        for label, ghost in zip(["b", "p", "i", "c"], ghosts):
            gx, gy = self._get_cell(ghost.rect)
            dist = hypot(gx - px, gy - py)
            ghost_feats[f"ghost_{label}_x"] = gx
            ghost_feats[f"ghost_{label}_y"] = gy
            ghost_feats[f"ghost_{label}_dist"] = round(dist, 4)
            ghost_feats[f"ghost_{label}_is_dangerous"] = int(ghost.state in DANGEROUS)
        seeds_grid = seeds_obj._SeedContainer__seeds
        seeds_remaining = seeds_obj._SeedContainer__seeds_counts
        seeds_eaten_count = TOTAL_SEEDS - seeds_remaining
        ndx, ndy, ndist = self._nearest_seed(px, py, seeds_grid)
        sr, sl, sd, su = self._directional_seeds(px, py, seeds_grid)
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
        seeds_ratio = round(seeds_eaten_count / TOTAL_SEEDS, 6)
        return {
            "pacman_x": px, "pacman_y": py,
            "moving_dir": moving_dir,
            "moving_axis": moving_axis,
            "wall_right": wall_right, "wall_left": wall_left,
            "is_reversing_right": is_reversing_right,
            "is_reversing_left": is_reversing_left,
            "is_reversing_up": is_reversing_up,
            "is_reversing_down": is_reversing_down,
            "wall_up": wall_up, "wall_down": wall_down,
            "lives_remaining": lives,
            **ghost_feats,
            "nearest_seed_dx": ndx, "nearest_seed_dy": ndy,
            "nearest_seed_dist": round(ndist, 4),
            "seeds_right": sr, "seeds_left": sl,
            "seeds_down": sd, "seeds_up": su,
            "nearest_energizer_dx": en_dx, "nearest_energizer_dy": en_dy,
            "nearest_energizer_dist": round(en_dist, 4),
            "energizers_remaining": en_remaining,
            "fruit_active": fruit_active,
            "seeds_eaten_ratio": seeds_ratio,
        }

    @staticmethod
    def _nearest_seed(px: int, py: int, seeds_grid: list) -> tuple:
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

        right = count(range(px, px + 5), range(py - 2, py + 3))
        left = count(range(px - 5, px),     range(py - 2, py + 3))
        down = count(range(px - 2, px + 3), range(py, py + 5))
        up = count(range(px - 2, px + 3), range(py - 5, py))
        return right, left, down, up
