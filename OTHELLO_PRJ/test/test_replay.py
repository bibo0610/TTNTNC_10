from pathlib import Path
from history.game_replay import GameReplay


def main():
    # Xác định thư mục gốc của project
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "game_logs"

    # Tìm các file JSON lịch sử
    logs = sorted(
        logs_dir.glob("game_*.json"),
        key=lambda p: p.stat().st_mtime
    )

    if not logs:
        print("Chua co file log.")
        print("Hay chay test_history.py truoc.")
        return

    # Chọn file lịch sử mới nhất
    filepath = logs[-1]
    print("Dang doc:", filepath)

    try:
        replay = GameReplay(filepath)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print("ERROR: Khong the doc lich su:", exc)
        return

    print("\nPASS: Du lieu lich su hop le!")
    print("Tong so luot:", len(replay.moves))

    # Hiển thị bàn cờ ban đầu
    replay.show()

    # Điều khiển Replay
    while True:
        command = input(
            "\n[n] Next | [p] Previous | "
            "[f] Final | [q] Quit: "
        ).strip().lower()

        if command == "n":
            replay.next()

        elif command == "p":
            replay.previous()

        elif command == "f":
            replay.go_to(len(replay.moves))

        elif command == "q":
            print("Da thoat Game Replay.")
            break

        else:
            print("Lenh khong hop le.")


if __name__ == "__main__":
    main()
