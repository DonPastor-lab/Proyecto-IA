import gradio as gr
import requests

from ollama_service import clean_input, make_decision

API_URL = "http://localhost:8000/predict"

def predict(user_input):

    try:
        # limpiar y estructurar la entrada
        cleaned_data = clean_input(user_input)

        # enviar los datos estructurados a FastAPI
        response = requests.post(
            API_URL,
            json=cleaned_data,
            timeout=30
        )

        if response.status_code != 200:
            print("Status:", response.status_code)
            print("Detalle del error:", response.text)

        response.raise_for_status()

        prediction = response.json()

        decision_explanation = make_decision(prediction["risk_code"])

        # mostrar datos y predicción
        return cleaned_data, prediction, decision_explanation

    except requests.RequestException as e:
        return {"error": str(e)}, None, None

    except Exception as e:
        return {"error": str(e)}, None, None


def load_template():
    template = '''
            1. edad: 
            2. estado civil:
            3. numero de dependientes económicos: 
            4. nivel educativo
            5. tipo de empleo:        
            6. años de antiguedad laboral:
            7. tipo de vivienda:
            8. ingreso mensual
            9. monto del prestamo:
            10. plazo en meses: 
            11. tasa de interes anual:
            12. cuota mensual:
            13. score crediticio:
            14. numero de lineas de credito:
            15. utilizacion de credito:
            16. deuda vigente:
            17. ratio deuda-ingreso:
            18. ratio prestamo-ingreso anual:
            19. pagos atrasados en los ultimos 12 meses:
            20. consultas de credito en los ultimos 6 meses:
            21. meses de historial crediticio:
            22. tiene cuenta de ahorro (1 si tiene, 0 si no):
            23. proposito del prestamo:
        '''
    return template


# crear interfaz
with gr.Blocks(title="Predicción de riesgo crediticio con Machine Learning") as demo:

    gr.Markdown("# Riesgo Crediticio")

    with gr.Row() as template_button:
            gr.Markdown("Ingrese la información del cliente",  scale=4)
            button_load = gr.Button("Cargar plantilla", scale=0.5)
    
    user_input = gr.Textbox(
        label="Descripción",
        placeholder="El cliente tiene 56 años...",
        lines=4
    )

    button_load.click(
        fn=load_template,
        outputs=user_input
    )

    button = gr.Button("Realizar predicción")

    decision_output = gr.Textbox(
        label="Respuesta del agente",
        lines=5,
        interactive=False
    )

    cleaned_output = gr.JSON(
        label="Datos procesados por Ollama"
    )

    prediction_output = gr.JSON(
        label="Resultado del modelo ML"
    )

    button.click(
        fn=predict,
        inputs=user_input,
        outputs=[cleaned_output, prediction_output, decision_output]
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)