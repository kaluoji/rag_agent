from openai import OpenAI
import sys

# Imprime un mensaje al inicio para confirmar que el script se está ejecutando
print("Iniciando prueba de modelo...")

try:
    # Muestra parcialmente la API key para verificación
    api_key = "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"
    print(f"Usando API key: {api_key[:10]}...{api_key[-10:]}")
    
    client = OpenAI(api_key=api_key)
    print("Cliente OpenAI creado")
    
    model_name = "ft:gpt-4o-mini-2024-07-18:personal::BCAkAcVL"
    print(f"Intentando acceder al modelo: {model_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Prueba de acceso al modelo"}
        ]
    )
    
    print("¡Éxito! Respuesta:", response.choices[0].message.content)
    
except Exception as e:
    # Captura cualquier error y muestra detalles completos
    print("Error encontrado:", e)
    print("Tipo de error:", type(e))
    import traceback
    traceback.print_exc()

print("Prueba finalizada")