import os
import time
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Importaciones específicas de RAGAS para testset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# Para manejar errores de OpenAI
from openai import RateLimitError, APIError, APITimeoutError

load_dotenv()
# Configuración
TEST_DOCS_PATH = "./test_docs"  # Ruta a la carpeta con los documentos
OUTPUT_PATH = "./testset"       # Ruta de salida para el testset
TESTSET_SIZE = 100                # Número de preguntas a generar
BATCH_SIZE = 20              # Procesar documentos en lotes más pequeños

# Asegurar que el directorio de salida exista
os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    start_time = time.time()
    
    # Verificar la clave API de OpenAI
    if "OPENAI_API_KEY" not in os.environ:
        api_key = input("Introduce tu clave API de OpenAI: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    print("Cargando documentos...")
    # Cargar documentos con loaders separados para cada tipo de archivo
    loaders = [
        DirectoryLoader(TEST_DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(TEST_DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error al cargar documentos: {e}")
    print(f"Cargados {len(docs)} documentos")
    
    if not docs:
        print("No se encontraron documentos en la carpeta. Verifica la ruta.")
        return
    
    # Dividir documentos en lotes más pequeños si hay muchos
    if len(docs) > BATCH_SIZE:
        print(f"Dividiendo {len(docs)} documentos en lotes de {BATCH_SIZE} para evitar límites de tasa")
        batches = [docs[i:i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    else:
        batches = [docs]  # Un solo lote con todos los documentos
    
    print("Configurando modelos LLM y embeddings...")
    # Configurar el modelo LLM y Embedding con los wrappers de RAGAS
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    embeddings = OpenAIEmbeddings()
    
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    print("Inicializando generador de testset...")
    # Inicializar el generador de testset
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
   
    # Dividir el número total de preguntas entre los lotes
    questions_per_batch = max(1, TESTSET_SIZE // len(batches))
    
    # Lista para almacenar todos los datasets generados
    all_datasets = []
    
    for i, batch in enumerate(batches):
        print(f"Procesando lote {i+1}/{len(batches)} con {len(batch)} documentos")
        
        # Ajustar el número de preguntas para el último lote
        if i == len(batches) - 1:
            questions = TESTSET_SIZE - (questions_per_batch * i)
        else:
            questions = questions_per_batch
            
        print(f"Generando {questions} preguntas para este lote...")
        
        # Intentar generar con reintentos en caso de límite de tasa
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generar testset para este lote
                dataset = generator.generate_with_langchain_docs(
                    batch, 
                    testset_size=questions
                )
                all_datasets.append(dataset)
                break  # Salir del bucle si tiene éxito
                
            except (RateLimitError, APIError, APITimeoutError) as e:
                retry_count += 1
                # Estrategia de backoff exponencial
                wait_time = 2 ** retry_count + 5  # 7, 9, 13, 21, 37 segundos
                
                if "retry_after" in str(e).lower():
                    # Intentar extraer el tiempo de espera recomendado
                    try:
                        # Esta es una extracción simple, se puede mejorar
                        import re
                        wait_suggestion = re.search(r'try again in (\d+\.\d+)s', str(e))
                        if wait_suggestion:
                            suggested_wait = float(wait_suggestion.group(1))
                            wait_time = suggested_wait + 1  # Añadir un segundo extra
                    except:
                        pass  # Usar el tiempo calculado por backoff
                
                print(f"Límite de tasa alcanzado. Reintento {retry_count}/{max_retries} después de {wait_time:.2f} segundos...")
                time.sleep(wait_time)
                
                if retry_count == max_retries:
                    print(f"No se pudo procesar el lote después de {max_retries} intentos. Continuando con el siguiente lote.")
            
            except Exception as e:
                print(f"Error inesperado: {e}")
                print("Continuando con el siguiente lote...")
                break
        
        # Pausa entre lotes para evitar alcanzar el límite de tasa
        if i < len(batches) - 1:
            pause_time = 10
            print(f"Pausa de {pause_time} segundos antes del siguiente lote...")
            time.sleep(pause_time)
    
    # Combinar todos los datasets
    if not all_datasets:
        print("No se pudo generar ningún conjunto de datos.")
        return
    
    # Para simplificar, usamos el primer dataset y agregamos el resto
    # Nota: Esto depende de la estructura interna de los datasets de RAGAS
    combined_df = pd.concat([dataset.to_pandas() for dataset in all_datasets if dataset])
    
    print("Guardando testset combinado...")
    # Guardar en CSV
    combined_df.to_csv(f"{OUTPUT_PATH}/testset.csv", index=False)
    
    # Guardar en JSON
    combined_df.to_json(f"{OUTPUT_PATH}/testset.json", orient="records", indent=2)
    
    print(f"Testset guardado en {OUTPUT_PATH}")
    print(f"Total de preguntas generadas: {len(combined_df)}")
    
    # Opcional: Subir a la plataforma RAGAS
    # Nota: esto solo funcionará si RAGAS permite subir dataframes directamente
    # Si no, necesitarás adaptarlo a tu caso
    upload_to_ragas = input("¿Quieres subir el testset a app.ragas.io? (s/n): ").lower() == 's'
    if upload_to_ragas:
        if "RAGAS_APP_TOKEN" not in os.environ:
            app_token = input("Introduce tu token de RAGAS: ")
            os.environ["RAGAS_APP_TOKEN"] = app_token
        
        try:
            # Si RAGAS necesita el objeto original, podrías necesitar ajustar esto
            all_datasets[0].upload()  # Asumiendo que podemos usar el primer dataset
            print("Dataset subido exitosamente a app.ragas.io")
        except Exception as e:
            print(f"Error al subir el dataset: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Proceso completado en {elapsed_time/60:.2f} minutos.")

if __name__ == "__main__":
    main()