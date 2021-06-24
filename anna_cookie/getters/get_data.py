import pandas as pd
import json
from anna_cookie import PROJECT_DIR


def get_skill() -> "DataFrame":
    """[summary]
    The function read the csv file into DataFrame

    Returns:
        [DataFrame]: Return a dataframe
    """
    filepath = PROJECT_DIR / "skills_en.csv"
    print(filepath)
    data = pd.read_csv(filepath)
    return data


def get_occupation() -> "DataFrame":
    """[summary]
    The function read the csv file into DataFrame

    Returns:
        [DataFrame]: Return a dataframe
    """
    filepath = PROJECT_DIR / "occupations_en.csv"
    data = pd.read_csv(filepath)
    return data


def get_ESCO() -> "json":
    """The function reads the ESCO_occup_skills.json file

    Returns:
        [json file]: [json file]
    """
    filepath = PROJECT_DIR / "ESCO_occup_skills.json"
    with open(filepath, "r") as stream:
        try:
            data = json.load(stream)
        except ValueError:
            print("no file found")
    return data
