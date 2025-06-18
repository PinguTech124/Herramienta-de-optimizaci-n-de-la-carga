# Herramienta-de-optimizaci-n-de-la-carga

# Guía de Instalación del Simulador EV con Anaconda + Streamlit

## 🧩 Requisitos

- Conexión a internet
- Windows, macOS o Linux

---

## 1️⃣ Descargar e Instalar Anaconda

1. Ve a [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
2. Haz clic en “Download” y elige tu sistema operativo
3. Descarga el instalador para Python 3.x (recomendado 3.10 o superior)
4. Ejecuta el instalador y sigue los pasos (puedes dejar todas las opciones por defecto)

---

## 2️⃣ Crear el entorno de trabajo

Una vez instalado Anaconda:

1. Abre **Anaconda Prompt** (en Windows) o una terminal
2. Navega a la carpeta donde se encuentre `environment.yml` (ejemplo: `cd Downloads`)
3. Ejecuta:

```
conda env create -f environment.yml
```

4. Cuando termine, activa el entorno con:

```
conda activate ev_parking_streamlit
```

---

## 3️⃣ Ejecutar la aplicación Streamlit

1. Asegúrate de estar en el entorno (`ev_parking_streamlit`)
2. Navega a la carpeta donde esté el proyecto
3. Ejecuta:

```
streamlit run app/app_streamlit.py
```

La app se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 📌 Notas

- Para salir de Streamlit: pulsa `Ctrl+C` en la terminal
- Para desactivar el entorno Conda: `conda deactivate`
