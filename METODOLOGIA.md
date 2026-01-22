# Metodología - Análisis de Sentimientos en Reviews de WhatsApp Business

## 1. Exploración de Datos y Limpieza

### 1.1 Carga y Exploración Inicial
- Descarga del dataset desde Kaggle: WhatsApp Business Reviews App Store
- Análisis exploratorio de datos (EDA) para identificar estructura, tipos de datos y valores faltantes
- Selección de variables relevantes: `version`, `score`, `title`, `text`, `country`

### 1.2 Limpieza de Texto
- **Detección de idioma**: Utilización de la librería `langdetect` para identificar el idioma de títulos y textos
- **Filtrado de idioma**: Se seleccionan únicamente reviews en inglés para garantizar consistencia en el análisis
- **Normalización de caracteres**: Eliminación de caracteres especiales, manteniendo solo letras y números
- **Limpieza de espacios**: Remoción de saltos de línea, tabuladores y espacios duplicados
- **Remoción de emojis**: Utilización de la librería `emoji` para eliminar emojis del texto

## 2. Procesamiento de Lenguaje Natural (NLP)

### 2.1 Análisis de Sentimiento con VADER
- **Herramienta**: SentimentIntensityAnalyzer de NLTK (Valence Aware Dictionary and sEntiment Reasoner)
- **Proceso**:
  - Análisis del título y texto del review por separado
  - Generación de scores de sentimiento: negatividad (`neg`), neutralidad (`neu`), positividad (`pos`) y sentimiento compuesto (`compound`)
  - El score `compound` varía de -1 (muy negativo) a +1 (muy positivo)

### 2.2 Categorización de Sentimientos
- **Umbrales de clasificación**:
  - Positivo: `compound > 0.05`
  - Negativo: `compound < -0.05`
  - Neutral: `-0.05 ≤ compound ≤ 0.05`
- **Variables generadas**:
  - `sentiment_text`: Clasificación del sentimiento del texto del review
  - `sentiment_title`: Clasificación del sentimiento del título
  - `compound_avg`: Promedio del sentimiento compuesto entre título y texto

### 2.3 Tokenización y Lematización
- **Tokenización**: División del texto en palabras utilizando `word_tokenize` de NLTK
- **Remoción de stopwords**: Eliminación de palabras comunes en inglés que no aportan significado (conjunciones, preposiciones, etc.)
- **Lematización**: Conversión de palabras a su forma raíz usando `WordNetLemmatizer`:
  - Lematización de verbos (pos='v')
  - Lematización de sustantivos (pos='n')

### 2.4 Extracción de Características de Texto
- **CountVectorizer**: Transformación del texto procesado en matriz de frecuencias de palabras
  - `max_features`: Limitación del número de características (máximo 50% del número de observaciones)
  - `min_df=5`: Palabras que aparecen en al menos 5 documentos
  - `max_df=0.7`: Palabras que aparecen en máximo el 70% de los documentos
- **Generación de variables**: Frecuencia de palabras clave que aportan al poder predictivo

## 3. Ingeniería de Características

### 3.1 Creación de Variables Objetivo
- **Variable categórica (dicotómica)**: `score_cat`
  - `1`: Reviews con score 4 o 5 (evaluación positiva)
  - `0`: Reviews con score 1, 2 o 3 (evaluación negativa/neutral)
- **Variable continua**: `score`
  - Valores originales del 1 al 5

### 3.2 Variables Categóricas
- **Normalización**: Agrupación de categorías con frecuencia < 5% en la categoría "Others"
- **Variables procesadas**: `country`, `version`, `sentiment_text`, `sentiment_title`
- **Codificación**: Transformación a variables dummy usando `pd.get_dummies()` con `drop_first=True`

### 3.3 Características Finales
- **Variables numéricas (18)**: Métricas de sentimiento (neg_text, neu_text, pos_text, compound_text, neg_title, neu_title, pos_title, compound_title, compound_avg) + variables dummy categóricas
- **Variables de texto (N)**: Frecuencia de palabras clave extraídas mediante CountVectorizer

## 4. Modelado Predictivo

### 4.1 División de Datos
- **Train-Test Split**: División 80-20 con `random_state=42`
- Separación independiente para variable categórica y continua
- Conjuntos guardados en archivos CSV para reproducibilidad

### 4.2 Regresión Logística para Variable Categórica

#### 4.2.1 Escalamiento de Datos
Utilización de tres escaladores diferentes para normalizar características:
- **StandardScaler**: Normalización a media=0 y desviación estándar=1
- **MinMaxScaler**: Escalado al rango [0, 1]
- **RobustScaler**: Escalado basado en cuartiles (robusto a outliers)

#### 4.2.2 Configuración del Modelo
- **Algoritmo**: Regresión Logística (sklearn.linear_model.LogisticRegression)
- **Hiperparámetros optimizados**:
  - `penalty`: Regularización ['l1', 'l2']
  - `solver`: ['liblinear']
  - `max_iter`: [100, 200, 500]

#### 4.2.3 Optimización de Hiperparámetros
- **Técnica**: GridSearchCV con validación cruzada (cv=4)
- **Métrica de optimización**: ROC-AUC (área bajo la curva ROC)
- **Pipeline**: Combinación automática de escalador + modelo para evitar data leakage
- **Paralelización**: Uso de `n_jobs=-1` para acelerar búsqueda

### 4.3 Validación y Evaluación

#### 4.3.1 Métricas de Clasificación (Variable Categórica)
- **ROC-AUC**: Área bajo la curva ROC (métrica principal durante entrenamiento)
- **Predicción en Test**: Evaluación del mejor modelo en conjunto de prueba
- **Probabilidades predichas**: Obtención de probabilidades de clase para interpretación

#### 4.3.2 Interpretación del Modelo
- **Coeficientes (Betas)**: Análisis de impacto de cada característica en la predicción
  - Coeficientes positivos: Aumentan probabilidad de score alto (4-5)
  - Coeficientes negativos: Disminuyen probabilidad de score alto
- **Términos independiente (Intercept)**: Probabilidad base del modelo

#### 4.3.3 Análisis de Variables Predictivas
- **Características numéricas de sentimiento**:
  - Variables que contribuyen negativamente/positivamente a la clasificación
  - Relevancia de sentimientos en título vs. texto del review
- **Palabras clave predictivas**:
  - Palabras asociadas a reviews negativos (ej: "limit", "code", "fix", "delete")
  - Palabras asociadas a reviews positivos (ej: "touch", "color", "click")

## 5. Consideraciones Metodológicas

### 5.1 Validez del Análisis de Sentimientos
- Correlación moderada entre VADER compound score y score original de reviews
- Reconocimiento de limitaciones de VADER: Mejor desempeño en textos más largos

### 5.2 Robustez del Modelado
- Uso de pipelines para garantizar que escalamiento solo se ajuste con datos de entrenamiento
- Validación cruzada para evitar sobreajuste
- Guardado de modelos entrenados en formato pickle para reproducibilidad

### 5.3 Interpretabilidad
- Preferencia por Regresión Logística por su interpretabilidad sobre modelos complejos
- Análisis directo de coeficientes para entender impacto de variables
- Separación de análisis entre características numéricas y palabras clave

## 6. Flujo General del Proyecto

```
1. DATOS CRUDOS
   ↓
2. LIMPIEZA Y DETECCIÓN DE IDIOMA
   ↓
3. ANÁLISIS DE SENTIMIENTO CON VADER
   ↓
4. PROCESAMIENTO DE TEXTO (Tokenización, Lematización)
   ↓
5. EXTRACCIÓN DE CARACTERÍSTICAS (CountVectorizer)
   ↓
6. INGENIERÍA DE FEATURES (Variables objetivo, encoding categóricas)
   ↓
7. DIVISIÓN TRAIN-TEST
   ↓
8. OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV
   ↓
9. ENTRENAMIENTO Y VALIDACIÓN DE REGRESIÓN LOGÍSTICA
   ↓
10. ANÁLISIS E INTERPRETACIÓN DE RESULTADOS
```

## 7. Librerías Principales Utilizadas

| Librería | Función |
|----------|---------|
| pandas | Manipulación de datos |
| NLTK | NLP: tokenización, lematización, stopwords, VADER |
| scikit-learn | Machine Learning: modelos, preprocesamiento, métricas |
| langdetect | Detección de idioma |
| emoji | Remoción de emojis |
| seaborn/matplotlib | Visualización |

---

**Última actualización**: Enero 2026
**Equipo**: Equipo 1 - Diplomado CDC M2 P2
