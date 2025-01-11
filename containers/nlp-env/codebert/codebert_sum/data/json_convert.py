import pandas as pd

for name in ["test", "train", "valid"]:
    with open(f"rare_{name}.json", encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile, lines=True)
    df.to_csv(f'{name}.csv', encoding='utf-8', index=False)
    df.head(1)[1]