import matplotlib.pyplot as plt
from numpy import mean
import pandas as pd
from pandas.core.arrays import categorical
from pandas.util.version import parse
from slugify import slugify

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def _parse_ingredients(s: str) -> tuple[int, list[str]]:
    if s == 'Unknown':
        return 0, []
    
    len, items = s.split('-')
    len = int(len)
    items = [i.strip() for i in items.split(',')]
    return len, items

def parse_ingredients(s: str) -> list[str]:
    _, items = _parse_ingredients(s)
    return items





if __name__ == '__main__':

    source_df =  pd.read_csv('./chocolate.csv')

    # normalize column name
    source_df.columns = [ slugify(col) for col in source_df.columns ]

    # drop empty columns if any
    source_df = source_df.dropna(axis=0, how='all')

    df = pd.DataFrame()

    # fill null entries if any
    df['ref'] = source_df['ref'].fillna(0)
    df['company-manufacturer'] = source_df['company-manufacturer'].fillna('Unknown')
    df['company-location'] = source_df['company-manufacturer'].fillna('Unknown')
    df['country-of-bean-origin'] = source_df['company-manufacturer'].fillna('Unknown')
    df['review-date'] = source_df['review-date'].fillna( source_df['review-date'].median())
    df['specific-bean-origin-or-bar-name'] = source_df['specific-bean-origin-or-bar-name'].fillna('Unknown')

    df['cocoa-percent'] = source_df['cocoa-percent'].map(lambda x: float(x.strip('%')) / 100)
    df['cocoa-percent'] = df['cocoa-percent'].fillna(df['cocoa-percent'].mean())

    df['ingredients'] = source_df['ingredients'].fillna('Unknown')

    df['ingredients'] = df['ingredients'].map(parse_ingredients)

    categorical_columns = ['company-manufacturer', 'company-location', 'country-of-bean-origin', 'specific-bean-origin-or-bar-name']
    label_encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    mlb = MultiLabelBinarizer()
    mlb_encoded = pd.DataFrame(mlb.fit_transform(df['ingredients']), columns=mlb.classes_)
    for col in mlb_encoded:
        df[f'ingredients_{col}'] = mlb_encoded[col]

    print(df)















