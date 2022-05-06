import pytest
import project3
import pandas as pd
import numpy as np

def test_dataframe():

    filename = "Unredactor.txt"

    df = project3.create_dataframe(filename)

    assert len(df) >0

