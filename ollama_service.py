import requests
import json
import os

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

# funcion para limpiar la entrada
def clean_input(user_input):

    prompt = f"""
    Tu tarea es extraer información de una solicitud de crédito.

Debes identificar los valores correspondientes a las variables
del JSON que aparece al final.

REGLAS IMPORTANTES:
1. Utiliza exactamente los nombres de las variables del JSON.
2. No cambies, traduzcas ni renombres las claves.
3. Extrae los valores que aparecen explícitamente en el texto.
4. No inventes información.
5. No coloques null si el dato aparece en el texto.
6. Si el dato no aparece, utiliza null.
7. Los números deben ser valores numéricos, no cadenas de texto.
8. Devuelve exclusivamente el JSON.

TEXTO DEL USUARIO:
{user_input}

JSON:
{{
    "edad": null,
    "estado_civil": null,
    "dependientes": null,
    "nivel_educacion": null,
    "tipo_empleo": null,
    "antiguedad_laboral_anios": null,
    "tipo_vivienda": null,
    "ingreso_mensual": null,
    "monto_prestamo": null,
    "plazo_meses": null,
    "tasa_interes_anual": null,
    "cuota_mensual": null,
    "score_crediticio": null,
    "num_lineas_credito": null,
    "utilizacion_credito": null,
    "deuda_vigente": null,
    "ratio_deuda_ingreso": null,
    "ratio_prestamo_ingreso_anual": null,
    "pagos_atrasados_12m": null,
    "consultas_credito_6m": null,
    "historial_crediticio_meses": null,
    "tiene_cuenta_ahorro": null,
    "proposito_prestamo": null
    }}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    generated_text = result.get("response", "")

    if not generated_text.strip():
        raise ValueError("Ollama devolvió una respuesta vacía")

    return json.loads(generated_text)


# funcion para que ollama explique el resultado
def make_decision(risk):

    if risk == 2:
        decision = "Rechazar solicitud"

    elif risk == 1:
        decision = "Solicitar documentación adicional"

    elif risk == 0:
        decision = "Aprobar solicitud"

    else:
        return "Riesgo no identificado"

    prompt = f"""
    El modelo de Machine Learning determinó:
    Nivel de riesgo: {risk}

    Decisión establecida: {decision}

    Reglas:
    - Riesgo 2 (alto): recomendar educación financiera.
    - Riesgo 1 (medio): solicitar documentación adicional
      y evaluar nuevamente.
    - Riesgo 0 (bajo): aprobar con condiciones estándar.

    Explica al usuario la decisión y su recomendación, no expliques las reglas, solo explica si fue aprobada (Riesgo 0),
    si se debe solicitar documentación adicional (Riesgo 1), o rechazar la solicitud y recomendar educación financiera (Riesgo 2).
    """

    response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
    
    response.raise_for_status()

    result = response.json()

    return result["response"]