import argparse
import pickle
from pprint import pprint

BASE_PATH = "/content/drive/MyDrive/tgn_results"

parser = argparse.ArgumentParser(description="Load TGN results by aggregator type")
parser.add_argument(
    "--agg",
    type=str,
    required=True,
    help="Aggregator type (e.g., mean, last, attention, weighted)"
)

args = parser.parse_args()

# build filename
file_path = f"{BASE_PATH}/tgn_{args.agg}.pkl"

print("Loading:", file_path)

with open(file_path, "rb") as f:
    data = pickle.load(f)

print("Object type:", type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())

print("\nFull contents:")
pprint(data)