"""
Generate synthetic first-24h ICU data for baseline experiments
"""
import numpy as np
import pandas as pd
import os

FEATURES = [
    'label', 'age', 'sex', 'hr', 'rr', 'map', 'spo2', 'temp', 'gcs_min', 'urine_24h',
    'wbc', 'hgb', 'platelets', 'na', 'k', 'bicarbonate', 'creatinine', 'bilirubin',
    'lactate', 'admission_type', 'icu_type'
]

N = 500  # rows
np.random.seed(42)

sexes = ['F', 'M']
admission_types = ['EMERGENCY', 'ELECTIVE', 'URGENT']
icu_types = ['MICU', 'SICU', 'CCU', 'TSICU']

def generate_row():
    return [
        np.random.binomial(1, 0.15),  # label
        np.random.randint(18, 90),    # age
        np.random.choice(sexes),      # sex
        np.random.randint(50, 140),   # hr
        np.random.randint(10, 40),    # rr
        np.random.randint(50, 120),   # map
        np.random.randint(85, 100),   # spo2
        np.random.uniform(35, 40),    # temp
        np.random.randint(3, 15),     # gcs_min
        np.random.uniform(0, 4000),   # urine_24h
        np.random.uniform(2, 20),     # wbc
        np.random.uniform(7, 17),     # hgb
        np.random.uniform(50, 400),   # platelets
        np.random.uniform(130, 150),  # na
        np.random.uniform(3.0, 5.5),  # k
        np.random.uniform(15, 30),    # bicarbonate
        np.random.uniform(0.5, 5.0),  # creatinine
        np.random.uniform(0.1, 10.0), # bilirubin
        np.random.uniform(0.5, 10.0), # lactate
        np.random.choice(admission_types),
        np.random.choice(icu_types)
    ]

data = [generate_row() for _ in range(N)]
df = pd.DataFrame(data, columns=FEATURES)
os.makedirs('data/synthetic', exist_ok=True)
df.to_csv('data/synthetic/first24h_minimal.csv', index=False)
