# ¿Llegará tarde este pedido? Predicción de retrasos en Olist

Proyecto final de Fundamentos de Aprendizaje Automático. Una herramienta para el equipo de operaciones de Olist (marketplace brasileño): **en el momento en que el vendedor entrega el paquete a la transportadora, estima la probabilidad de que llegue después de la fecha prometida al cliente** y explica por qué.

- **Datos:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, CC BY-NC-SA 4.0): ~100 mil pedidos reales de 2016–2018 en 9 tablas.
- **Paradigmas:** no supervisado (segmentación de vendedores con K-Means, clustering jerárquico, DBSCAN y PCA) → supervisado (clasificación con kNN, árbol de decisión, Random Forest, Gradient Boosting y SVM, explicada con SHAP).

## Resultados principales

| | Valor |
|---|---|
| Modelo elegido | Gradient Boosting (`HistGradientBoostingClassifier`) |
| PR-AUC en prueba (may–ago 2018, nunca vista) | **0.39** (azar: 0.044) |
| ROC-AUC en prueba | **0.82** (estable mes a mes: 0.80–0.85) |
| Revisando el 10 % de despachos de mayor riesgo cada día | 1 de cada 5 marcados llega tarde (5× el azar); se detecta el 48 % de los retrasos |
| Segmentos de vendedores | 5: Lentos y con quejas, Alto volumen, Baratos con flete alto, Nacionales, Locales ágiles |

Decisiones clave, cada una justificada con evidencia en el notebook:

- Validación **temporal** (entrenar con el pasado, evaluar con el futuro), porque la tasa de retrasos cambia en el tiempo.
- El segmento del vendedor **no mejora** el clasificador de forma consistente: se usa como herramienta de negocio en la interfaz, no como variable del modelo.
- Ni `class_weight`, ni submuestreo, ni **SMOTENC**, ni **CTGAN** mejoran la PR-AUC, y los dos primeros distorsionan las probabilidades: no se balancea.
- Gradient Boosting y SVM empatan; se elige Gradient Boosting porque entrega probabilidades y explicaciones SHAP exactas.
- Las alertas se generan como **top 10 % de cada día**, que respeta la capacidad del equipo aunque los puntajes deriven (Olist acortó sus promesas de entrega a mediados de 2018).

## Estructura

```
Final Boss/
├── model_development.ipynb   # paso a paso: EDA, segmentación, modelos, generalización, métricas, SHAP, reajuste
├── src/olist_ml.py           # transformaciones compartidas por el notebook y la interfaz
├── app/app.py                # interfaz Streamlit para el equipo de operaciones
├── app/requirements.txt      # dependencias mínimas para desplegar solo la interfaz
├── models/late_delivery_model.joblib   # modelo evaluado + umbrales + segmentador + foto de vendedores
├── requirements.txt          # dependencias completas
└── Requeriments_FinalP.pdf   # enunciado
```

| Requisito del enunciado | Dónde |
|---|---|
| 3.1 Problema | Notebook §1 |
| 3.2 Preprocesamiento | §3, §4, §7 |
| 3.3 EDA e insights | §5 (con tabla insight → decisión en §5.11) |
| 3.4 Paradigmas | §8 no supervisado, §9 supervisado |
| 3.5 Metodología y selección | §9, §12 |
| 3.6 Generalización | §6, §10 |
| 3.7 Métricas | §11 |
| 3.8 Despliegue | §13 y `app/app.py` |
| 3.9 Reajuste | §14 |

## Cómo ejecutar

```bash
pip install -r requirements.txt
```

**Notebook:** abrir `model_development.ipynb` y ejecutar todas las celdas desde la carpeta `Final Boss`. Los datos se descargan automáticamente del dataset oficial de Kaggle con `kagglehub` (no requiere token). La ejecución completa tarda ~12 minutos en CPU y regenera `models/late_delivery_model.joblib`.

**Interfaz:**

```bash
streamlit run app/app.py
```

Se abre en `http://localhost:8501`, con cuatro pestañas:

- **Evaluar un despacho:** formulario del pedido → probabilidad, semáforo, acción sugerida, segmento del vendedor y razones (SHAP).
- **Lote del día:** subir un CSV con los despachos del día y descargar la lista priorizada.
- **Ejemplos reales:** pedidos del periodo de prueba con su resultado real.
- **Segmentos de vendedores.**

**Publicarla en internet:** en [Streamlit Community Cloud](https://streamlit.io/cloud), conectar este repositorio y elegir `Final Boss/app/app.py` como archivo principal. Se instala `app/requirements.txt`; el modelo ya va en `models/` y no se necesitan los datos crudos.

## Limitaciones

El dataset no incluye la transportadora ni eventos de rastreo, probablemente la mayor fuente de información sobre retrasos, y termina en 2018. Las probabilidades se sobreestiman cuando el periodo tiene menos retrasos que el entrenamiento; por eso la lista diaria usa el orden de riesgo y no un corte fijo.
