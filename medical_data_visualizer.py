import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar los datos
df = pd.read_csv('medical_examination.csv')

# 2. Agregar columna 'overweight' (Sobrepeso)
# El BMI se calcula: peso (kg) / altura (metros) al cuadrado.
# OJO: La altura en el dataset está en centímetros, hay que dividir por 100.
# Si BMI > 25 es 1 (sobrepeso), si no 0.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalizar datos
# Normalizamos 'cholesterol' y 'gluc'. Si es 1 (normal) se vuelve 0 (bueno).
# Si es > 1 (malo) se vuelve 1 (malo).
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. Función para dibujar el Gráfico de Categorías
def draw_cat_plot():
    # 5. Crear DataFrame para el cat plot usando pd.melt
    # 'melt' transforma la tabla de formato ancho a formato largo.
    # Mantenemos 'cardio' fijo y convertimos las otras columnas en filas.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupar y reformatear
    # Contamos cuántas veces aparece cada valor (0 o 1) para cada variable, separado por si tienen cardio o no.
    # Es obligatorio renombrar la columna de conteo a 'total' para que pase los tests.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7. Dibujar el gráfico con sns.catplot
    # kind='bar' crea barras. col='cardio' crea dos gráficos separados (uno para sanos, otro para enfermos).
    g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    
    # 8. Obtener la figura para el output
    fig = g.fig

    # No modificar las siguientes dos líneas
    fig.savefig('catplot.png')
    return fig


# 10. Función para dibujar el Mapa de Calor
def draw_heat_map():
    # 11. Limpiar los datos
    # Filtramos datos incorrectos o extremos (segmentos de pacientes erróneos)
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & # La presión baja no puede ser mayor que la alta
        (df['height'] >= df['height'].quantile(0.025)) & # Altura menor al percentil 2.5
        (df['height'] <= df['height'].quantile(0.975)) & # Altura mayor al percentil 97.5
        (df['weight'] >= df['weight'].quantile(0.025)) & # Peso menor al percentil 2.5
        (df['weight'] <= df['weight'].quantile(0.975))   # Peso mayor al percentil 97.5
    ]

    # 12. Calcular la matriz de correlación
    corr = df_heat.corr()

    # 13. Generar una máscara para el triángulo superior
    # Un mapa de calor de correlación es simétrico (la correlación de A con B es igual a B con A).
    # Ocultamos la mitad superior para que se vea más limpio.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15. Dibujar el mapa de calor con sns.heatmap
    
    # fmt='.1f' asegura que veamos un decimal. vmax=.3 ajusta el contraste de colores para pasar el test.
    sns.heatmap(corr, mask=mask, vmax=.3, center=0, annot=True, fmt='.1f', square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # No modificar las siguientes dos líneas
    fig.savefig('heatmap.png')
    return fig