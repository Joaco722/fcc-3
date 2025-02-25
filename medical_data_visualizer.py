import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
# Definir una función para aplicar la condición
def is_overweight(row):
    bmi = row['weight'] / ((row['height'] / 100) ** 2)
    return 1 if bmi > 25 else 0

df['overweight'] = df.apply(is_overweight, axis=1)

# 3
def normalize_cholesterol(row):
    return 0 if (row['cholesterol']) == 1 else 1

def normalize_gluc(row):
    return 0 if (row['gluc']) == 1 else 1

df['cholesterol'] = df.apply(normalize_cholesterol, axis=1)
df['gluc'] = df.apply(normalize_gluc, axis=1)


# 12
def draw_heat_map():
    # Limpiar los datos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(12, 9))

    # Dibujar el mapa de calor utilizando seaborn
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', linewidths=0.5, ax=ax)

    fig.savefig('heatmap.png')
    return fig



