# **Trabajo Práctico Integrador - Aprendizaje Autómatico I

## Integrantes:
-  Fontana, Gustavo
-  Menescaldi, Brisa

## **Predicción de Condiciones Climáticas en Ciudades de Australia**

Este repositorio contiene el desarrollo de un proyecto enfocado en la predicción de la lluvia para ciudades seleccionadas de Australia utilizando técnicas de aprendizaje automático y redes neuronales. El objetivo principal es explorar y comparar varios modelos de regresión y clasificación para predecir las variables 'RainTomorrow' (si lloverá mañana) y 'RainfallTomorrow' (cantidad de lluvia) basado en datos climáticos históricos.

## **Objetivos**
El trabajo tiene como objetivos principales:

-  Utilizar la biblioteca scikit-learn para preprocesar datos, implementar modelos de regresión y clasificación, y evaluar métricas.

-  Utilizar TensorFlow para entrenar redes neuronales y comparar su desempeño con los modelos clásicos.

-  Utilizar Streamlit para desplegar la aplicación de predicción del modelo seleccionado.

-  Realizar un análisis exploratorio detallado de los datos climáticos proporcionados, enfocándose en la comprensión de las variables y su comportamiento.

-  Optimizar la selección de hiperparámetros mediante técnicas como grid search, random search u optuna.

-  Implementar métodos de explicabilidad del modelo, como SHAP, para entender las decisiones del modelo a nivel local y global.

## **Contenido del Repositorio**

1-  **Notebook de Jupyter**: Contiene el desarrollo del proyecto, incluyendo el análisis exploratorio, la implementación de modelos y la evaluación de métricas.

2-  **Carpeta de Scripts**:
-  **Funciones y Modelos**: Scripts que contienen funciones para preprocesamiento de datos y definición de modelos de regresión y clasificación.
-  **Despliegue de Aplicación**: Scripts para desplegar la aplicación de predicción utilizando Streamlit para cada modelo evaluado.

3-  **Dataset**: El conjunto de datos utilizado, denominado weatherAUS.csv, que contiene registros climáticos de varias ciudades australianas.

## **Contenido Detallado**

El trabajo se estructura en las siguientes secciones principales:

-  Análisis Exploratorio: Exploración detallada de las variables, manejo de datos faltantes, visualización mediante histogramas, scatterplots y diagramas de caja, y evaluación del balance del dataset.

-  Modelos de Regresión: Implementación y evaluación de modelos como regresión lineal, métodos de gradiente descendiente y regularización (Lasso, Ridge, Elastic Net).

-  Modelos de Clasificación: Implementación y evaluación de regresión logística, métricas como precisión, recall, F1 Score y visualización de matrices de confusión y curvas ROC.

-  Optimización de Hiperparámetros: Uso de técnicas como grid search y optuna para optimizar los hiperparámetros de los modelos seleccionados.

-  Explicabilidad del Modelo: Uso de SHAP para entender las características más influyentes en las predicciones de los modelos.

-  Redes Neuronales: Implementación y comparación de modelos de redes neuronales con los modelos tradicionales de regresión y clasificación.

-  Comparación de Modelos: Evaluación comparativa de todos los modelos utilizados para determinar el mejor rendimiento según la métrica seleccionada.

-  MLOps: Desarrollo de scripts utilizando Streamlit para la puesta en producción del modelo seleccionado, permitiendo la predicción de condiciones climáticas futuras.

## **Requisitos para desplegar el contenido del repositorio**:
-  gdown==5.2.0
-  imbalanced-learn==0.12.3
-  matplotlib==3.9.0
-  numpy==1.26.4
-  optuna==3.6.1
-  pandas==2.2.2
-  seaborn==0.13.2
-  scikit-learn==1.5.0
-  scipy==1.13.1
-  shap==0.45.1
-  tensorflow==2.16.1
