import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy


class GameHistory:
    def __init__(self):
        self.data = {
            "game": "Othello 8x8",
            "created_at": datetime.now().isoformat(),
            "initial_board": None,
            "moves": [],
            "result": None,
        }

    def start(self, board):
        self.data["initial_board"] = deepcopy(board)

    def record(self, player, action, board):
        self.data["moves"].append({
            "turn": len(self.data["moves"]) + 1,
            "player": player,
            "action": list(action) if action is not None else "PASS",
            "board": deepcopy(board),
        })

    def finish(self, winner):
        self.data["result"] = {
            "winner": winner
        }

    def save(self, filename=None):
        folder = Path("game_logs")
        folder.mkdir(exist_ok=True)

        if filename is None:
            filename = (
                "game_" +
                datetime.now().strftime("%Y%m%d_%H%M%S_%f") +
                ".json"
            )

        path = folder / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.data,
                f,
                ensure_ascii=False,
                indent=2
            )

        return path
