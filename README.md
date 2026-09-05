# Laboratorio de Algoritmos Genéticos (Machine Learning)

> **Asignatura:** Machine Learning — Semestre VIII  
> **Autores:** Erik Arevalo & Oscar Duque  
> **Lenguaje:** Python 3.10+  
> **Librerías principales:** NumPy, Matplotlib  

## Descripción General

Este proyecto implementa un marco modular y extensible para **Algoritmos Genéticos Canónicos (AG)** basado en el paradigma de **Programación Orientada a Objetos (POO)**.

## Estructura del Proyecto

```text
Ejercicios algoritmos geneticos/
│
├── main.py                          # Menú principal interactivo en consola
├── requirements.txt                 # Dependencias del entorno virtual (numpy, matplotlib)
├── README.md                        # Documentación completa del proyecto
├── .gitignore                       # Filtro de archivos para control de versiones Git
│
└── src/                             # Paquete con la lógica modular
    ├── __init__.py                  # Inicializador del paquete src
    ├── AlgoritmoGenetico.py         # Clase base con el bucle evolutivo genérico
    ├── MaximizacionCubica.py        # Ejercicio 1: Maximización f(x) = x³ - 4x² + 5x
    ├── OptimizaciónMultivariable.py # Ejercicio 2: Minimización f(x,y) = x² + y²
    ├── ImpactoPM.py                 # Ejercicio 3: Análisis de sensibilidad con pm = {0.01, 0.1, 0.5}
    └── ElitismoAmpliado.py          # Ejercicio 4: Elitismo con preservación de los K=3 mejores
```

---

## Instalación y Configuración

### 1. Prerrequisitos
- Tener instalado **Python 3.10** o superior en el sistema.
- Terminal de comandos (**PowerShell**, **CMD** o **Bash**).

### 2. Creación del Entorno Virtual
Desde la raíz del proyecto, ejecuta:

```powershell
python -m venv venv
```

### 3. Activación del Entorno Virtual

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Si los scripts están bloqueados por directiva, ejecuta antes: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **Windows (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **Linux / macOS / Git Bash:**
  ```bash
  source venv/bin/activate
  ```

### 4. Instalación de Dependencias
Con el entorno virtual activado, instala las librerías necesarias:

```powershell
pip install -r requirements.txt
```

O instalándolas directamente:
```powershell
pip install numpy matplotlib
```

---

## Ejecución del Programa

Para iniciar el menú interactivo, ejecuta:

```powershell
python main.py
```