# ¿Llegará tarde este pedido? Predicción de retrasos en Olist

Proyecto final de Fundamentos de Aprendizaje Automático. Una herramienta para el equipo de operaciones de Olist (marketplace brasileño): **en el momento en que el vendedor entrega el paquete a la transportadora, estima la probabilidad de que llegue después de la fecha prometida al cliente** y explica por qué.

## Integrantes y datos académicos

| Integrante | Identificación |
|---|---|
| Sebastian Salazar Henao | 1022003377 |
| Andres Felipe Velez Alvarez | 1021923619 |
| Samuel Samper Cardona | 1040180930 |
| Nathalia Cardoza | 4992531 |

- **Universidad:** EAFIT.
- **Docente:** Yomin Estiven Jaramillo Munera.
- **Curso:** Fundamentos de Aprendizaje Automático.
- **Fecha de entrega:** 3 de Noviembre de 2026.

## Enlaces de entrega

| Entregable | Enlace o ubicación |
|---|---|
| Repositorio | [FinalProject-MachineLearning](https://github.com/AndresVelez31/FinalProject-MachineLearning) |
| Aplicación desplegada | https://olist-alerta-retrasos.streamlit.app/ |
| Video de demostración de 2 minutos | **PENDIENTE** |
| Notebook de desarrollo | [model_development.ipynb](model_development.ipynb) |

## Datos y enfoque

- **Datos:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, CC BY-NC-SA 4.0): aproximadamente 100.000 pedidos de 2016-2018 en nueve tablas.
- **Usuarios:** equipo de operaciones y servicio al cliente de Olist.
- **Decisión que apoya:** priorizar seguimiento de despachos y contacto con clientes ante posibles retrasos.
- **Variable objetivo:** `late = 1` si el día real de entrega es posterior a la fecha prometida; `0` en caso contrario.
- **Aprendizaje no supervisado:** segmentación de vendedores con K-Means, contrastada con clustering jerárquico, DBSCAN y PCA.
- **Aprendizaje supervisado:** comparación de kNN, árbol de decisión, Random Forest, Gradient Boosting y SVM, con explicaciones mediante SHAP.

## Resultados de la versión actual

| | Valor |
|---|---|
| Modelo elegido | Gradient Boosting (`HistGradientBoostingClassifier`) |
| PR-AUC en prueba (may–ago 2018) | **0.3941** (azar: aproximadamente 0.044) |
| ROC-AUC en prueba | **0.8238** (rango mensual reportado: 0.80–0.85) |
| Política del notebook de aproximadamente 10 % diario | Precisión aproximada: 21.8 %; recall aproximado: 48.5 % |
| Segmentos de vendedores | 5: Lentos y con quejas, Alto volumen, Baratos con flete alto, Nacionales, Locales ágiles |

Decisiones documentadas en los experimentos actuales del notebook:

- Validación **temporal** (entrenar con el pasado, evaluar con el futuro), porque la tasa de retrasos cambia en el tiempo.
- El segmento del vendedor **no mejora** el clasificador de forma consistente: se usa como herramienta de negocio en la interfaz, no como variable del modelo.
- Ni `class_weight`, ni submuestreo, ni **SMOTENC**, ni **CTGAN** mejoran la PR-AUC, y los dos primeros distorsionan las probabilidades: no se balancea.
- Gradient Boosting y SVM obtienen resultados cercanos; se elige Gradient Boosting por sus probabilidades y la posibilidad de utilizar `TreeExplainer` para las explicaciones SHAP.
- La política operativa propuesta prioriza aproximadamente el **10 % de mayor riesgo de cada día**. Queda pendiente unificar su implementación entre el notebook y la aplicación, incluidos redondeo, desempates y archivos que contengan varios días.

## Estructura

```
FinalProject-MachineLearning/
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
| 3.6 Generalización | §6, §10; pendientes de corrección temporal al final de este README |
| 3.7 Métricas | §11 |
| 3.8 Despliegue | §13, `app/app.py` y enlace público pendiente |
| 3.9 Reajuste | §14 |

## Cómo ejecutar

Ejecutar los comandos desde la carpeta raíz del repositorio, preferiblemente en un entorno virtual. La aplicación se comprobó durante la revisión con Python 3.12 y `scikit-learn==1.9.0`.

**Descargar el proyecto:**

```bash
git clone https://github.com/AndresVelez31/FinalProject-MachineLearning.git
cd FinalProject-MachineLearning
```

**Ejecutar solamente la aplicación:**

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/app.py
```

Abrir la dirección mostrada en la terminal, normalmente `http://localhost:8501`. Debe estar presente `models/late_delivery_model.joblib`. Mantener la versión fijada de scikit-learn para reproducir el entorno del modelo guardado.

La interfaz contiene cuatro pestañas:

- **Evaluar un despacho:** formulario del pedido, probabilidad, semáforo, acción sugerida, segmento del vendedor y razones mediante SHAP.
- **Lote del día:** carga de CSV y descarga de la lista priorizada. Incluye validaciones básicas; revisar los pendientes de robustez indicados al final.
- **Ejemplos reales:** pedidos del periodo de prueba con su resultado observado.
- **Segmentos de vendedores:** perfiles operativos identificados con K-Means.

**Ejecutar el desarrollo completo:**

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook model_development.ipynb
```

Abrir el notebook y ejecutar todas las celdas en orden desde la raíz del repositorio. Los datos se descargan desde Kaggle con `kagglehub`; la primera descarga requiere acceso a internet. El tiempo registrado en la ejecución guardada es de aproximadamente 11.4 minutos y varía según el equipo. El notebook regenera el archivo del modelo.

## Estrategia de reajuste

La sección 14 del notebook propone monitorear deriva mediante PSI y revisar el desempeño cuando se conozcan los resultados de entrega. Contempla evaluación mensual del reentrenamiento del clasificador, actualización trimestral de la segmentación, comparación del candidato con el modelo vigente y conservación de una versión anterior para permitir una reversión.

El notebook incluye una simulación del reajuste. La automatización operativa del mantenimiento se presenta como estrategia propuesta, no como un servicio ya implementado.

## Limitaciones

- El dataset termina en 2018; estos resultados no demuestran desempeño sobre operaciones actuales.
- No se dispone de identificación de la transportadora ni de eventos de rastreo.
- La evaluación utiliza pedidos entregados con información suficiente; no cubre por sí sola cancelaciones, pérdidas o todos los pedidos todavía en tránsito.
- Las probabilidades pueden sobreestimarse cuando cambia la frecuencia de retrasos. Recalcular los umbrales del semáforo no equivale a calibrar las probabilidades.
- La segmentación tiene separación limitada: los grupos se interpretan como perfiles operativos.
- La validación temporal y la consistencia de la interfaz requieren cerrar los pendientes siguientes.

## Fuente de los datos

Olist. *Brazilian E-Commerce Public Dataset by Olist*. Disponible en [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). La utilización de los datos debe respetar las condiciones de su licencia.
