import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from itertools import combinations, product
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, kruskal
from sklearn.preprocessing import LabelEncoder

# Configuración de Streamlit
st.set_page_config(page_title="EDA Regresión - Lab 3", layout="wide", initial_sidebar_state="expanded")
sns.set_style('whitegrid')

# ==================== FUNCIONES EXACTAS DEL NOTEBOOK (Adaptadas a Streamlit) ====================

def encontrar_csv(root=Path.cwd()):
    ruta_csv = root / 'data' / 'raw' / 'dataset_Regresión.csv'
    if ruta_csv.exists():
        return ruta_csv
    elif root != root.parent:
        root = root.parent
        return encontrar_csv(root)
    else:
        raise FileNotFoundError("No se encontró el archivo dataset_Regresión.csv")

def indentificar_tipos_de_variables(df,umbral_categoricas=8):
    categoricas = []
    numericas = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique()  <= umbral_categoricas and df[col].nunique() <len(df)*0.05:
                categoricas.append(col)
            else:
                numericas.append(col)
        else:
            categoricas.append(col)
    
    return categoricas, numericas

def detectar_outliers(df, variable):
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[variable]  < lower_bound) | (df[variable]  > upper_bound)]
    return outliers

def crear_histogramas_sin_outliers(df, columna, bins=None, color="skyblue", titulo=None, clip_outliers=True):
    data = df[columna].dropna()

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    data_sin_outliers = data[(data  >= lower_bound)  & (data  <= upper_bound)]
    n_outliers = len(data) - len(data_sin_outliers)

    if clip_outliers and len(data_sin_outliers)  >= max(3, len(data) * 0.5):
        data_plot = data_sin_outliers
        rango_text = f"(sin outliers: {lower_bound:.2f} a {upper_bound:.2f})"
    else:
        data_plot = data
        rango_text =  "(incluye todos los valores)"

    if bins is None:
        if data_plot.nunique() <= 20:
            bins = data_plot.nunique()
        else:
            q75p, q25p = np.percentile(data_plot, [75, 25])
            iqrp = q75p - q25p
            if iqrp  > 0:
                bin_width = 2 * iqrp / (len(data_plot) ** (1/3))
                bins = int(np.ceil((data_plot.max() - data_plot.min()) / bin_width))
                bins = max(15, min(bins, 40))
            else:
                bins = 20

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    ax.hist(data_plot, bins=bins, color=color, edgecolor='black', alpha=0.85)

    if titulo is None:
        titulo = f"Distribución de {columna} {rango_text}"
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xlabel(columna, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)

    Mediana = data.median()
    Promedio = data.mean()
    Varianza = data.var()

    ax.axvline(Mediana, color='red', linestyle='dashed', linewidth=1.5, label=f'Mediana: {Mediana:.2f}')
    ax.axvline(Promedio, color='green', linestyle='dashed', linewidth=1.5, label=f'Promedio: {Promedio:.2f}')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)

    stats_text = (
        f'n = {len(data)}\n'
        f'Media = {Promedio:.2f}\n'
        f'Mediana = {Mediana:.2f}\n'
        f'Varianza = {Varianza:.2f}\n'
        f'Rango datos = {data.min():.2f} - {data.max():.2f}\n'
        f'Valores usados = {len(data_plot)} ({n_outliers} outliers excluidos)'
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def crear_histogramas_con_outliers(df, columna, bins=None, color="skyblue", titulo=None):
    data = df[columna].dropna()

    if bins is None:
        if data.nunique() <= 20:
            bins = data.nunique()
        else:
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr  > 0:
                bin_width = 2 * iqr / (len(data) ** (1/3))
                bins = int(np.ceil((data.max() - data.min()) / bin_width))
                bins = max(15, min(bins, 40))
            else:
                bins = 20

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.85)

    if titulo is None:
        titulo = f"Distribución de {columna}"
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    ax.set_xlabel(columna, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)

    Mediana = data.median()
    Promedio = data.mean()
    Varianza = data.var()

    ax.axvline(Mediana, color='red', linestyle='dashed', linewidth=1.5, label=f'Mediana: {Mediana:.2f}')
    ax.axvline(Promedio, color='green', linestyle='dashed', linewidth=1.5, label=f'Promedio: {Promedio:.2f}')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=10)

    stats_text = f'n = {len(data)}\nMedia = {Promedio:.2f}\nMediana = {Mediana:.2f}\nVarianza = {Varianza:.2f}\nRango: {data.min():.2f} - {data.max():.2f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

# ==================== PRUEBAS ESTADÍSTICAS ====================

def obtener_resultados_chi(df, cate, alpha=0.05, filtro="Todas"):
    resultados = []
    for var1, var2 in combinations(cate, 2):
        tabla = pd.crosstab(df[var1], df[var2])
        chi2, p, dof, expected = chi2_contingency(tabla)
        dependencia = p < alpha
        
        if filtro == "Dependencia" and not dependencia: continue
        if filtro == "Independencia" and dependencia: continue
            
        resultados.append({
            'Variable 1': var1, 'Variable 2': var2,
            'Estadístico χ²': f"{chi2:.4f}", 'p-valor': f"{p:.4f}", 'gl': dof,
            'Conclusión': "Existe evidencia de dependencia" if dependencia else "No hay evidencia suficiente de dependencia"
        })
    return pd.DataFrame(resultados)

def obtener_resultados_spearman(df, num, alpha=0.05, filtro="Todas"):
    resultados = []
    for var1, var2 in combinations(num, 2):
        coef, p = spearmanr(df[var1], df[var2])
        fuerza = "Débil" if abs(coef) < 0.3 else ("Moderada" if abs(coef) < 0.7 else "Fuerte")
        direccion = "Positiva" if coef > 0 else "Negativa"
        significativa = p < alpha
        
        if filtro == "Asociación" and not significativa: continue
        if filtro == "Sin Asociación" and significativa: continue
            
        resultados.append({
            'Variable 1': var1, 'Variable 2': var2,
            'Coeficiente': f"{round(abs(coef),5)}", 'Fuerza': fuerza, 'Dirección': direccion,
            'p-valor': f"{p:.4f}", 
            'Conclusión': "Existe evidencia de asociación monótona significativa" if significativa else "No hay evidencia suficiente de asociación"
        })
    return pd.DataFrame(resultados)

def obtener_resultados_kruskal(df, num, cate, alpha=0.05, filtro="Todas"):
    resultados = []
    n_obs = len(df)
    for var1, var2 in product(num, cate):
        grupos = [df[var1][df[var2] == cat] for cat in df[var2].unique()]
        if len(grupos) < 2: continue
        k = len(grupos)
        h_stat, p = kruskal(*grupos)
        epsilon_sq = max(0, (h_stat - k + 1) / (n_obs - k))
        fuerza = "Despreciable" if epsilon_sq < 0.01 else ("Pequeño" if epsilon_sq < 0.08 else ("Moderada" if epsilon_sq < 0.26 else "Grande"))
        significativa = p < alpha
        
        if filtro == "Diferencias" and not significativa: continue
        if filtro == "Sin Diferencias" and significativa: continue
            
        conclusion = (f"La distribución de '{var1}' varía significativamente entre al menos dos categorías de '{var2}'" 
                      if significativa else 
                      f"No hay evidencia suficiente para afirmar diferencias en la distribución de '{var1}' entre las categorías de '{var2}'")
        
        resultados.append({
            'Variable Numérica': var1, 'Variable Categórica': var2,
            'H_estadístico': f"{h_stat:.4f}", 'Tamaño del efecto': fuerza, 'p-valor': f"{p:.4f}",
            'Conclusión': conclusion
        })
    return pd.DataFrame(resultados)

# ==================== INTERFAZ STREAMLIT ====================

def main():
    st.title("📊 Análisis Exploratorio de Datos - Regresión")
    st.markdown("---")

    # 1. CARGA DE DATOS
    st.header("1. Carga de Datos")
    if st.button("Cargar Dataset"):
        try:
            ruta_csv = encontrar_csv()
            df = pd.read_csv(ruta_csv)
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
            st.session_state['df_original'] = df
            st.success(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.error(f"Error al cargar: {e}")

    if 'df_original' not in st.session_state:
        st.stop()

    df = st.session_state['df_original']

    # 2. CORRECCIÓN DE DATOS INCONSISTENTES
    st.header("2. Corrección de Datos Inconsistentes")
    if st.button("Aplicar Corrección de Valores"):
        columnas = list(df.columns)
        
        # Valores inconsistentes o erróneos en las columnas
        valor1 = df[columnas[25]].unique()[2]
        valor2 = df[columnas[26]].unique()[135]
        valor3 = df[columnas[28]].unique()[161]
        
        # Reemplazo
        df[columnas[25]] = df[columnas[25]].replace(valor1, "No ")
        df[columnas[26]] = df[columnas[26]].replace(valor2, 1.42)
        df[columnas[28]] = df[columnas[28]].replace(valor3, 3.1)
        
        st.session_state['df_corregido'] = df
        st.success("✅ Corrección aplicada exitosamente")
        st.write(f"**Valores corregidos:**")
        st.write(f"- Columna {columnas[25]}: {valor1} → 'No '")
        st.write(f"- Columna {columnas[26]}: {valor2} → 1.42")
        st.write(f"- Columna {columnas[28]}: {valor3} → 3.1")

    if 'df_corregido' in st.session_state:
        df = st.session_state['df_corregido']
    else:
        st.warning("⚠️ Aplica la corrección de datos primero")
        st.stop()

    # 3. IMPUTACIÓN
    st.header("3. Tratamiento de Datos Faltantes")
    st.subheader("Valores faltantes originales")
    st.write(df.isna().sum()[df.isna().sum() > 0])

    cols_numericas = df.select_dtypes(include='number').columns
    df_media = df.fillna(df.mean(numeric_only=True))
    imputer_knn = KNNImputer(n_neighbors=5)
    df_numeric_imputed = pd.DataFrame(imputer_knn.fit_transform(df[cols_numericas]), columns=cols_numericas, index=df.index)
    df_knn = df.copy()
    df_knn[cols_numericas] = df_numeric_imputed

    metodo = st.radio("Seleccione método de imputación:", ["Original (sin imputar)", "Media", "KNN"], horizontal=True)
    
    if metodo == "Original":
        df_tratado = df.copy()
    elif metodo == "Media":
        df_tratado = df_media
    else:
        df_tratado = df_knn

    st.success(f"Método seleccionado: {metodo}")
    st.subheader("Estadísticas después de imputación")
    st.dataframe(df_tratado.describe())

    st.session_state['df'] = df_tratado
    df = df_tratado

    # 4. IDENTIFICACIÓN DE VARIABLES
    categoricas, numericas = indentificar_tipos_de_variables(df)
    Valores_a_quitar_cat = ['Program', 'What are the skills do you have ?', 'What is you interested area?']
    categoricas_limpias = [var for var in categoricas if var not in Valores_a_quitar_cat]
    Variables_mas_importante_categoricas = categoricas_limpias
    Variables_mas_importante_numericas = numericas
    target = "What is your current CGPA?"

    st.header("4. Tipos de Variables")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Numéricas ({len(numericas)}):**")
        st.write(numericas)
    with col2:
        st.write(f"**Categóricas ({len(categoricas_limpias)}):**")
        st.write(categoricas_limpias)

    # 5. EDA VISUALIZACIONES
    st.header("5. Análisis Exploratorio & Visualizaciones")
    tab1, tab2, tab3, tab4 = st.tabs(["Categóricas", "Numéricas (Histogramas)", "Boxplots", "Scatter Plots"])

    with tab1:
        var_cat = st.selectbox("Variable categórica:", Variables_mas_importante_categoricas)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=var_cat, data=df.dropna(), ax=ax)
        plt.xticks(rotation=45)
        plt.title(f'Plot de {var_cat}')
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        var_num_hist = st.selectbox("Variable numérica:", Variables_mas_importante_numericas)
        incluir_outliers = st.checkbox("Incluir outliers en histograma", value=False)
        if incluir_outliers:
            fig = crear_histogramas_con_outliers(df, var_num_hist)
        else:
            fig = crear_histogramas_sin_outliers(df, var_num_hist)
        st.pyplot(fig)

    with tab3:
        st.write("Boxplots de CGPA por categoría:")
        var_cat_box = st.selectbox("Categoría para Boxplot:", Variables_mas_importante_categoricas, key="box_cat")
        temp_df = df[[var_cat_box, target]].dropna()
        if len(temp_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            temp_df.boxplot(column=target, by=var_cat_box, ax=ax)
            plt.suptitle('')
            plt.xlabel(var_cat_box, fontsize=12, fontweight='bold')
            plt.ylabel(target, fontsize=12, fontweight='bold')
            plt.title(f'Distribución de CGPA por {var_cat_box}', fontsize=13, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

    with tab4:
        var_num_scatter = st.selectbox("Variable X (Scatter):", [v for v in Variables_mas_importante_numericas if v != target])
        temp_df = df[[var_num_scatter, target]].dropna()
        if len(temp_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.scatter(temp_df[var_num_scatter], temp_df[target], alpha=0.6, edgecolors='k', s=70, color='steelblue')
            ax.set_title(f'Relación entre {var_num_scatter} y CGPA', fontsize=14, fontweight='bold')
            ax.set_xlabel(var_num_scatter, fontsize=12)
            ax.set_ylabel(target, fontsize=12)
            ax.grid(True, alpha=1)
            plt.tight_layout()
            st.pyplot(fig)

    # 6. CORRELACIONES
    st.header("6. Correlaciones")
    tipo_corr = st.selectbox("Método de correlación:", ["Spearman", "Kendall", "Pearson"])
    
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    corr_target = df_numeric.corr(method=tipo_corr.lower())[target].drop(target).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    corr_target.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title(f'Correlación ({tipo_corr}) de variables con {target}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Correlación', fontsize=12)
    ax.set_xlabel('Variables', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # 7. PRUEBAS ESTADÍSTICAS
    st.header("7. Pruebas de Asociación y Dependencia")
    alpha = st.slider("Nivel de significancia (α):", 0.01, 0.10, 0.05, 0.01)

    p_chi, p_spear, p_krusk = st.tabs(["Chi-Cuadrado", "Spearman", "Kruskal-Wallis"])

    with p_chi:
        st.subheader("Prueba Chi-Cuadrado (Categóricas)")
        filtro_chi = st.selectbox("Filtrar resultados:", ["Todas", "Dependencia", "Independencia"])
        if st.button("Ejecutar Chi-Cuadrado"):
            res_chi = obtener_resultados_chi(df, Variables_mas_importante_categoricas, alpha, filtro_chi)
            if not res_chi.empty:
                st.dataframe(res_chi, use_container_width=True)
                st.info(f"Total de pruebas: {len(obtener_resultados_chi(df, Variables_mas_importante_categoricas, alpha, 'Todas'))}")
            else:
                st.warning("No hay resultados con el filtro seleccionado.")

    with p_spear:
        st.subheader("Prueba de Spearman (Numéricas)")
        filtro_spear = st.selectbox("Filtrar resultados:", ["Todas", "Asociación", "Sin Asociación"], key="filtro_s")
        if st.button("Ejecutar Spearman"):
            res_spear = obtener_resultados_spearman(df, Variables_mas_importante_numericas, alpha, filtro_spear)
            if not res_spear.empty:
                st.dataframe(res_spear, use_container_width=True)
                st.info(f"Total de pares: {len(obtener_resultados_spearman(df, Variables_mas_importante_numericas, alpha, 'Todas'))}")
            else:
                st.warning("No hay resultados con el filtro seleccionado.")

    with p_krusk:
        st.subheader("Prueba de Kruskal-Wallis (Numérica vs Categórica)")
        filtro_krusk = st.selectbox("Filtrar resultados:", ["Todas", "Diferencias", "Sin Diferencias"], key="filtro_k")
        if st.button("Ejecutar Kruskal-Wallis"):
            res_krusk = obtener_resultados_kruskal(df, Variables_mas_importante_numericas, Variables_mas_importante_categoricas, alpha, filtro_krusk)
            if not res_krusk.empty:
                st.dataframe(res_krusk, use_container_width=True)
                st.info(f"Total de comparaciones: {len(obtener_resultados_kruskal(df, Variables_mas_importante_numericas, Variables_mas_importante_categoricas, alpha, 'Todas'))}")
            else:
                st.warning("No hay resultados con el filtro seleccionado.")

if __name__ == "__main__":
    main()