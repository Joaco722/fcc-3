import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)

df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6
    df_cat['value'] = df_cat['value'].astype(int)
    
    # 7
    # Ordenar los valores en la columna 'variable'
    df_cat['variable'] = pd.Categorical(df_cat['variable'], categories=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], ordered=True)
    
    # 8
    # Crear el gráfico categórico
    g = sns.catplot(x="variable", hue="value", col="cardio", data=df_cat, kind="count", height=5, aspect=1.2)
    
    # 9
    g.set_axis_labels("variable", "total")  # Actualizar la etiqueta del eje y

    # 10
    fig = g.fig

    # 11
    fig.savefig('catplot.png')
    return fig  # Asegurarse de devolver fig


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




