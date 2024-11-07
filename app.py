import pandas as pd
import plotly.express as px
from shiny import App, ui, render, reactive
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Cargar los archivos CSV
docentes_df = pd.read_csv("DOCENTES.csv")
alumnos_df = pd.read_csv("ALUMNOS.csv")

# Preparación del modelo con más parámetros
alumnos_df["PERTENECE AL TERCIO SUPERIOR"] = alumnos_df["PERTENECE AL TERCIO SUPERIOR"].map({"SI": 1, "NO": 0})
alumnos_df = alumnos_df.dropna(subset=["UBICACION RANKING FACULTAD", "CREDITOS APROBADOS", "FACULTAD", "PAIS NACIMIENTO"])

# Codificar las columnas de 'FACULTAD' y 'PAIS NACIMIENTO' a valores numéricos
facultad_encoder = LabelEncoder()
pais_encoder = LabelEncoder()
alumnos_df["FACULTAD_ENCODED"] = facultad_encoder.fit_transform(alumnos_df["FACULTAD"])
alumnos_df["PAIS_ENCODED"] = pais_encoder.fit_transform(alumnos_df["PAIS NACIMIENTO"])

# Seleccionar características para el modelo
X = alumnos_df[["UBICACION RANKING FACULTAD", "PERTENECE AL TERCIO SUPERIOR", "FACULTAD_ENCODED", "PAIS_ENCODED"]]
y = alumnos_df["CREDITOS APROBADOS"]

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Tamaño de página y carpeta para imágenes
page_size = 10
img_dir = Path("images")
img_dir.mkdir(exist_ok=True)

# Definir la interfaz de usuario
app_ui = ui.page_fluid(
    ui.h2("Dashboard con Paginación, Filtros, Gráficos y Predicción para Docentes y Alumnos"),
    
    # Sección para Docentes con filtro
    ui.h3("Datos de Docentes"),
    ui.input_select("filtro_facultad_docente", "Filtrar por Facultad", 
                    choices=["Todos"] + list(docentes_df["FACULTAD"].unique()), selected="Todos"),
    ui.input_slider("page_docentes", "Página de Docentes", min=1, max=(len(docentes_df) // page_size) + 1, value=1),
    ui.output_table("table_docentes"),
    ui.output_image("plot_docentes"),
    
    # Sección para Alumnos con filtros
    ui.h3("Datos de Alumnos"),
    ui.input_select("filtro_facultad_alumno", "Filtrar por Facultad", 
                    choices=["Todos"] + list(alumnos_df["FACULTAD"].unique()), selected="Todos"),
    ui.input_select("filtro_tercio", "Filtrar por Pertenencia al Tercio Superior", 
                    choices=["Todos", "SI", "NO"], selected="Todos"),
    ui.input_slider("page_alumnos", "Página de Alumnos", min=1, max=(len(alumnos_df) // page_size) + 1, value=1),
    ui.output_table("table_alumnos"),
    ui.output_image("plot_alumnos"),

    # Sección para predicción
    ui.h3("Predicción de Créditos Aprobados"),
    ui.input_numeric("ranking_input", "Ubicación en Ranking de Facultad", value=1000, min=1),
    ui.input_checkbox("tercio_input", "¿Pertenece al Tercio Superior?", value=True),
    ui.input_select("facultad_input", "Facultad", choices=list(facultad_encoder.classes_)),
    ui.input_select("pais_input", "País de Nacimiento", choices=list(pais_encoder.classes_)),
    ui.output_text("prediction_output")
)

# Definir la lógica del servidor
def server(input, output, session):
    # Filtro y paginación para docentes
    @output
    @render.table
    @reactive.event(input.page_docentes, input.filtro_facultad_docente)
    def table_docentes():
        # Filtrar por facultad seleccionada
        df = docentes_df
        if input.filtro_facultad_docente() != "Todos":
            df = df[df["FACULTAD"] == input.filtro_facultad_docente()]
        # Paginación
        start = (input.page_docentes() - 1) * page_size
        end = start + page_size
        return df.iloc[start:end]

    # Gráfico de pastel para la categoría de docentes
    @output
    @render.image
    def plot_docentes():
        fig = px.pie(docentes_df, names="CATEGORIA", title="Distribución de Docentes por Categoría",
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
        image_path = img_dir / "plot_docentes.png"
        fig.write_image(image_path, scale=2)
        return {"src": str(image_path), "height": "400px"}

    # Filtro y paginación para alumnos
    @output
    @render.table
    @reactive.event(input.page_alumnos, input.filtro_facultad_alumno, input.filtro_tercio)
    def table_alumnos():
        # Filtrar por facultad y pertenencia al tercio superior
        df = alumnos_df
        if input.filtro_facultad_alumno() != "Todos":
            df = df[df["FACULTAD"] == input.filtro_facultad_alumno()]
        if input.filtro_tercio() != "Todos":
            df = df[df["PERTENECE AL TERCIO SUPERIOR"] == (1 if input.filtro_tercio() == "SI" else 0)]
        # Paginación
        start = (input.page_alumnos() - 1) * page_size
        end = start + page_size
        return df.iloc[start:end]

    # Gráfico de pastel para alumnos por pertenencia al tercio superior
    @output
    @render.image
    def plot_alumnos():
        fig = px.pie(alumnos_df, names="PERTENECE AL TERCIO SUPERIOR", 
                     title="Proporción de Alumnos en el Tercio Superior",
                     color_discrete_map={"SI": "blue", "NO": "red"}, hole=0.3)
        image_path = img_dir / "plot_alumnos.png"
        fig.write_image(image_path, scale=2)
        return {"src": str(image_path), "height": "400px"}

    # Predicción de créditos aprobados
    @output
    @render.text
    def prediction_output():
        ranking_value = input.ranking_input()
        tercio_value = 1 if input.tercio_input() else 0
        facultad_value = facultad_encoder.transform([input.facultad_input()])[0]
        pais_value = pais_encoder.transform([input.pais_input()])[0]
        
        # Realizar la predicción
        prediction = model.predict([[ranking_value, tercio_value, facultad_value, pais_value]])[0]
        return f"Créditos Aprobados Estimados: {prediction:.2f}"

# Crear la aplicación
app = App(app_ui, server)
