from pathlib import Path

data = Path.home() / "data"

print(list(data.iterdir()))
