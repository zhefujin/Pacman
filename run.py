import argparse

from pygame import display, font, mixer, transform

from pacman import Game
from pacman.misc import load_image


def pg_setup():
    icon = transform.scale(load_image("ico"), (256, 256))
    font.init()
    mixer.init()
    display.init()
    display.set_icon(icon)
    display.set_caption("PACMAN")


def parse_args():
    parser = argparse.ArgumentParser(description="Pacman Game")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start the game with data recording enabled. Recorded data will be saved as CSV files in ./recordings/ directory.",
    )
    parser.add_argument(
        "--record-dir",
        default="recordings",
        help="Directory to save recorded CSV files (default: recordings). Only effective if --record is specified.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Let the trained AI model control Pacman autonomously.",
    )
    parser.add_argument(
        "--model-dir",
        default="assignment",
        help="Directory containing the trained model (default: assignment).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pg_setup()
    game = Game(
        record=args.record,
        record_dir=args.record_dir,
        autoplay=args.autoplay,
        model_dir=args.model_dir,
    )
    game.main_loop()


if __name__ == "__main__":
    main()
