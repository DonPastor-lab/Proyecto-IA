import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#Cargar el modelo entrenado y las columnas exactas del One-Hot Encoding
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "modelo"

try:
    modelo = joblib.load(MODELS_DIR / "modelo_riesgo.joblib")
    columnas_modelo = joblib.load(MODELS_DIR / "columnas_modelo.joblib")
except Exception as e:
    raise RuntimeError(f"Error al cargar los archivos del modelo en {MODELS_DIR}: {e}")

app = FastAPI(
    title="API de Evaluación de Riesgo Crediticio",
    description="Microservicio que expone el modelo de Árbol de Decisión mediante /predict",
    version="1.0.0"
)

# 2. Validación de entrada con Pydantic (las 23 características del dataset
class ClienteInput(BaseModel):
    edad: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    estado_civil: str = Field(..., description="Ej: Soltero, Casado, Divorciado, Viudo")
    dependientes: int = Field(..., ge=0, description="Número de dependientes económicos")
    nivel_educacion: str = Field(..., description="Ej: Secundaria, Técnico, Universitario, Posgrado")
    tipo_empleo: str = Field(..., description="Ej: Empleado, Independiente, Empresario, Jubilado")
    antiguedad_laboral_anios: float = Field(..., ge=0)
    tipo_vivienda: str = Field(..., description="Ej: Propia, Alquilada, Familiar, Hipotecada")
    ingreso_mensual: float = Field(..., gt=0)
    monto_prestamo: float = Field(..., gt=0)
    plazo_meses: int = Field(..., gt=0)
    tasa_interes_anual: float = Field(..., ge=0)
    cuota_mensual: float = Field(..., gt=0)
    score_crediticio: int = Field(..., ge=0, le=1000)
    num_lineas_credito: int = Field(..., ge=0)
    utilizacion_credito: float = Field(..., ge=0)
    deuda_vigente: float = Field(..., ge=0)
    ratio_deuda_ingreso: float = Field(..., ge=0)
    ratio_prestamo_ingreso_anual: float = Field(..., ge=0)
    pagos_atrasados_12m: int = Field(..., ge=0)
    consultas_credito_6m: int = Field(..., ge=0)
    historial_crediticio_meses: int = Field(..., ge=0)
    tiene_cuenta_ahorro: int = Field(..., ge=0, le=1, description="1 si tiene, 0 si no")
    proposito_prestamo: str = Field(..., description="Ej: Vivienda, Vehículo, Negocio, Personal")

# Mapeo según el enunciado: 0 -> low, 1 -> medium, 2 -> high
MAPEO_RIESGO = {
    0: "low",
    1: "medium",
    2: "high"
}

@app.get("/")
def health_check():
    return {"status": "online", "mensaje": "API de Riesgo Crediticio funcionando en perfecta forma"}

@app.post("/predict")
def predict(cliente: ClienteInput):
    try:
        # Convertir los datos validados por Pydantic a un DataFrame de 1 fila
        df_cliente = pd.DataFrame([cliente.model_dump()])

        # Aplicar One-Hot Encoding a las mismas 5 variables categóricas del entrenamiento
        variables_categoricas = [
            "estado_civil",
            "nivel_educacion",
            "tipo_empleo",
            "tipo_vivienda",
            "proposito_prestamo"
        ]
        df_encoded = pd.get_dummies(df_cliente, columns=variables_categoricas, dtype=int)

        # Alinear exactamente con las 41 columnas del modelo entrenado
        df_final = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

        # Realizar la predicción y obtener la probabilidad (score)
        clase_predicha = int(modelo.predict(df_final)[0])
        probabilidades = modelo.predict_proba(df_final)[0]
        score_confianza = round(float(max(probabilidades)), 4)

        # Extraer las 3 variables más importantes del árbol para ayudar a la justificación del Agente
        importancias = sorted(
            zip(columnas_modelo, modelo.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        factores_clave = [
            {"variable": var, "importancia": round(float(imp), 4)}
            for var, imp in importancias
        ]

        # Respuesta en el formato JSON
        return {
            "risk_level": MAPEO_RIESGO.get(clase_predicha, "unknown"),
            "risk_code": clase_predicha,
            "score": score_confianza,
            "top_features": factores_clave
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la predicción: {str(e)}")