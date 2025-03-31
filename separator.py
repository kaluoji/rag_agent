import pandas as pd

# Lee el CSV (ajusta el separador si es necesario, p.ej. sep=';' para CSV separados por punto y coma)
df = pd.read_csv('visa_mastercard_v5_rows.csv')

# Exporta a un archivo Excel (.xlsx)
df.to_excel('visa_mastercard_v5_rows.xlsx', index=False)

df.to_csv('visa_mastercard_v5_rows.csv', index=False)   
