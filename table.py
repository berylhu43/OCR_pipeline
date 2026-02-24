import pandas as pd
import os

pd.set_option('display.max_columns', None)

output_dir = 'output'


for img_dir in os.listdir(output_dir):
    img_path = os.path.join(output_dir, img_dir)
    mmd_path = os.path.join(img_path, "result.mmd")
    tables = pd.read_html(mmd_path)

    for i, df in enumerate(tables):
        print(f"Table {i}")
        print(df)

