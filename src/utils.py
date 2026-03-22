from pathlib import Path

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logger

def save_object(file_path, obj):
    try:
        path_obj = Path(file_path)

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logger.exception(f"Error occured in Save object {e}")
        raise