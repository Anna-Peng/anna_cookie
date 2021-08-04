import pandas as pd
import json
from anna_cookie import PROJECT_DIR


def get_skill() -> pd.DataFrame:
    """[summary]
    The function read the csv file into DataFrame

    Returns:
        [DataFrame]: Return a dataframe
    """
    filepath = PROJECT_DIR / "inputs" / "data" / "skills_en.csv"
    # print(filepath)
    data = pd.read_csv(filepath)
    return data


def get_occupation() -> pd.DataFrame:
    """[summary]
    The function read the csv file into DataFrame

    Returns:
        [DataFrame]: Return a dataframe
    """
    filepath = PROJECT_DIR / "inputs" / "data" / "occupations_en.csv"
    # print(filepath)
    data = pd.read_csv(filepath)
    return data


def get_ESCO() -> dict:
    """The function reads the ESCO_occup_skills.json file

    Returns:
        [json file]: [json file]
    """
    filepath = PROJECT_DIR / "inputs" / "data" / "ESCO_occup_skills.json"
    # print(filepath)
    with open(filepath, "r") as stream:
        try:
            data = json.load(stream)
        except ValueError:
            print("no file found")
    return data
