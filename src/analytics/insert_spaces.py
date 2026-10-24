# Adds the six virtual study spaces (Table XII of the proposal) to the database. Run once.
import os
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

load_dotenv()
spaces = pd.DataFrame([
    # space_id, name, source, capacity, use_type, quiet, outlets, x, y (position on the map)
    ("S1", "Library",            "ROBOD", 20,  "Individual",          1, 1, 2, 2),
    ("S2", "Large lecture room", "ROBOD", 40,  "Group",               0, 1, 6, 2),
    ("S3", "Small lecture room", "ROBOD", 24,  "Individual or group", 0, 0, 9, 2),
    ("S4", "Lecture room",       "SDU",   84, "Group",               0, 1, 2, 5),
    ("S5", "Study zone 1",       "SDU",   32, "Individual",          1, 1, 6, 5),
    ("S6", "Study zone 2",       "SDU",   32, "Group",               0, 0, 9, 5),
], columns=["space_id", "name", "source", "capacity", "use_type", "quiet", "outlets", "x", "y"])

engine = sa.create_engine(os.getenv("DB_URL"))
spaces.to_sql("spaces", engine, if_exists="append", index=False)
print(pd.read_sql("SELECT * FROM spaces", engine))