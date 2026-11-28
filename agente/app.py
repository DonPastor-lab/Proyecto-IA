import gradio as gr
import requests
import json

try:
    from agente.ollama_service import clean_input, make_decision
except ImportError:
    from ollama_service import clean_input, make_decision

API_URL = "http://localhost:8000/predict"


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


def extract(user_input):
    # limpiar y estructurar la entrada
    cleaned_data = clean_input(user_input)

    return json.dumps(cleaned_data, indent=2, ensure_ascii=False), gr.Tab(interactive=True), gr.Tabs(selected="step2")


# funcion para confirmar que los datos se extrajeron de forma correcta y proceder a la tab3
def confirm(editor):
    try:
        features = json.loads(editor) 
    except:
        raise gr.Error(f"El JSON no es válido: {e}")

    return features, features, gr.Tab(interactive=True), gr.Tabs(selected="step3")


def predict(features):
    try:
        # enviar los datos a FastAPI
        response = requests.post(
            API_URL,
            json=features,
            timeout=30
        )

        response.raise_for_status()
        prediction = response.json()

        decision_explanation = make_decision(prediction["risk_code"])

        return decision_explanation, json.dumps(prediction)
    
    except requests.RequestException as e:
        return {"error": str(e)}, None, None

    except Exception as e:
        return {"error": str(e)}, None, None


CSS = """
.fixed-box { height: 250px; overflow-y: auto; }
.centered-row { justify-content: center; }
.gradio-container { width: 60vw !important; margin: auto; }
"""

with gr.Blocks(title="Riesgo crediticio", css=CSS) as demo:
    gr.Markdown("# Evaluación de Riesgo Crediticio")
    
    features_state = gr.State({})

    with gr.Tabs(selected="step1") as tabs:

        with gr.Tab("1 - Datos", id="step1") as tab1:
            button_load = gr.Button("Plantilla", size="sm", scale=0)

            user_input = gr.Textbox(
                label="Datos del solicitante", 
                lines=15, 
                max_lines=20)
            
            btn_extraer = gr.Button("Extraer variables", variant="primary")

        with gr.Tab("2 - Revisar variables", id="step2", interactive=False) as tab2:
            gr.Markdown("¿Las variables son correctas? Puedes editar el JSON.")

            editor = gr.Code(
                language="json", 
                interactive=True, 
                lines=15)
            
            with gr.Row():
                btn_reextract = gr.Button("Volver a extraer")
                btn_confirm = gr.Button("Confirmar y continuar", variant="primary")

        with gr.Tab("3 - Predicción", id="step3", interactive=False) as tab3:
            preview = gr.JSON(
                label="Preview de las variables", 
                elem_classes="fixed-box"
                )
            
            btn_predecir = gr.Button("Predecir", variant="primary")

            # mostrar datos y predicción
            with gr.Accordion("Resultado del modelo"):
                prediction_output = gr.JSON(
                    elem_classes="fixed-box"
                )

            gr.Markdown("## Decisión")

            decision_output = gr.Markdown(
                label="Explicación",
                show_label=True,
                container=True, 
            )


    # eventos
    button_load.click(
        fn=load_template,
        outputs=user_input
    )

    btn_extraer.click(
        fn=extract, 
        inputs=user_input, 
        outputs=[editor, tab2, tabs],
        show_progress_on=user_input,
    )
    
    btn_reextract.click(
        fn=extract, 
        inputs=user_input, 
        outputs=[editor, tab2, tabs]
    )  # nueva petición a ollama

    btn_confirm.click(
        fn=confirm, 
        inputs=editor, 
        outputs=[features_state, preview, tab3, tabs])
    
    btn_predecir.click(
        fn=predict, 
        inputs=features_state, 
        outputs=[decision_output, prediction_output])

demo.launch()