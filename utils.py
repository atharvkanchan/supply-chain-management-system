import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    # normalize column names
    df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))

    # date parsing: try common names
    for col in ['date','shipment_date','created_at']:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col])
            break
    if 'date' not in df.colu

