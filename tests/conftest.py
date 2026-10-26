import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ["src/recommender", "src/forecasting", "app"]:
    sys.path.insert(0, os.path.join(ROOT, p))