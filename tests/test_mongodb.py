
import os
import sys

# Add NeuralShield project root to Python path
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, ROOT_DIR)

from backend.database.mongodb import test_connection


if __name__ == "__main__":

    print("\n========== NEURALSHIELD DATABASE TEST ==========")

    if test_connection():
        print("Database verification: PASSED")
    else:
        print("Database verification: FAILED")
