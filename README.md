# Documentación del Agente Regulatorio

## Introducción
El Agente Experto Regulatorio es una solución avanzada basada en inteligencia artificial diseñada para proporcionar información precisa y actualizada sobre normativas y regulaciones del sector financiero. El sistema utiliza tecnologías de Procesamiento de Lenguaje Natural (NLP) y Recuperación Aumentada por Generación (RAG) para ofrecer respuestas detalladas y generar informes normativos personalizados basados en documentación oficial.

## Propósito y Alcance
Este sistema está diseñado para:

Responder consultas específicas sobre normativas regulatorias
Generar informes normativos detallados en formato Word
Proporcionar análisis de riesgos e impactos de normativas
Facilitar el acceso a información regulatoria actualizada para profesionales del sector

## Tecnologías Clave
El sistema se basa en las siguientes tecnologías:

Backend: Python 3.10 con FastAPI
Base de datos vectorial: Supabase para almacenamiento de embeddings
Modelos de IA: Integración con modelos de lenguaje avanzados
Frontend: React.js con Material UI
Generación de documentos: Biblioteca docx para Python

### Diagrama General
┌─────────────────┐     ┌────────────────────────────────────┐
│                 │     │            BACKEND                 │
│    FRONTEND     │     │ ┌──────────────┐ ┌──────────────┐  │
│                 │     │ │              │ │              │  │
│  - React        │     │ │  API Layer   │ │   Agentes    │  │
│  - Material UI  │◄────┼─┤  (FastAPI)   │ │              │  │
│  - WebSockets   │     │ │              │ │              │  │
│                 │     │ └──────┬───────┘ └───────┬──────┘  │
└─────────────────┘     │        │                 │         │
                        │ ┌──────▼─────────────────▼──────┐  │
                        │ │                               │  │
                        │ │       Orquestador             │  │
                        │ │                               │  │
                        │ └───────────────┬───────────────┘  │
                        │                 │                  │
                        │ ┌───────────────▼───────────────┐  │
                        │ │                               │  │
                        │ │      Base de Conocimiento     │  │
                        │ │      (Supabase Vector DB)     │  │
                        │ │                               │  │
                        │ └───────────────────────────────┘  │
                        └────────────────────────────────────┘

### Componentes Principales
El sistema está compuesto por los siguientes componentes interconectados:

Frontend: Interfaz de usuario web construida con React
Backend API: Capa de servicios RESTful implementada con FastAPI
Sistema Multi-Agente: Conjunto de agentes especializados para diferentes tareas
Orquestador: Componente central que coordina los diferentes agentes
Base de Conocimiento Vectorial: Almacenamiento de documentos procesados con Supabase

## Arquitectura del Sistema

El proyecto se divide en dos componentes principales:

### Backend (Python/FastAPI)

El backend está desarrollado con FastAPI y Python, implementando un sistema multi-agente que coordina la interacción entre diversos módulos especializados:

1. **Agente Orquestador**: Coordina el flujo de trabajo entre los diferentes agentes, determinando qué agente debe responder a cada consulta.

2. **Agente de Comprensión de Consultas**: Analiza y entiende la intención del usuario, expandiendo la consulta y extrayendo entidades clave.

3. **Agente Experto Regulatorio**: Proporciona respuestas detalladas basadas en la normativa aplicable.

4. **Agente de Informes**: Genera documentos formales en formato Word con análisis detallados de normativas específicas.

### Frontend (React)

El frontend está desarrollado con React y utiliza Material UI para proporcionar una interfaz de usuario intuitiva y profesional:

1. **Interfaz de Chat**: Permite a los usuarios realizar consultas en lenguaje natural.

2. **Visualizador de Documentos**: Permite previsualizar, editar y anotar documentos generados.

3. **Sistema de Historial**: Mantiene un registro de consultas previas y reportes generados.

## Capacidades Principales

### Búsqueda Semántica y RAG

El sistema utiliza:

1. **Embeddings Vectoriales**: Para búsqueda semántica avanzada.

2. **Recuperación Aumentada con Generación (RAG)**: Combina la recuperación de documentación relevante con generación de texto para proporcionar respuestas precisas y fundamentadas.

3. **Re-ranking**: Mejora la relevancia de los resultados mediante análisis semántico profundo.

### Análisis de Consultas

Capacidades avanzadas de procesamiento de consultas:

1. **Expansión de Consultas**: Detecta términos implícitos para mejorar la recuperación.

2. **Detección de Entidades**: Identifica entidades relevantes como normas, procesos, requisitos, etc.

3. **Clasificación de Intenciones**: Determina si el usuario busca explicación, comparación, instrucciones, etc.

### Generación de Informes

Informes normativos con formato profesional que incluyen:

1. **Estructura Estandarizada**: Con secciones predefinidas (resumen ejecutivo, marco regulatorio, análisis, etc.).

2. **Formato DOCX**: Generación de documentos Word con formato profesional.

3. **Vista Previa HTML**: Capacidad de visualizar los informes en el navegador.

4. **Anotaciones y Edición**: Herramientas para edición colaborativa y anotaciones.

## Integración con Base de Datos

El sistema utiliza Supabase para:

1. **Base de Datos Vectorial**: Almacenamiento y búsqueda eficiente de embeddings.

2. **Almacenamiento de Documentos**: Gestión de reportes generados y metadatos asociados.

## Procesamiento de Documentos

Capacidades avanzadas para manejar documentos normativos:

1. **OCR Avanzado**: Extracción de texto de documentos escaneados.

2. **Chunking Semántico**: División inteligente de documentos en fragmentos coherentes.

3. **Metadatos Enriquecidos**: Categorización, extracción de palabras clave, etc.

## API y Endpoints Principales

El servidor expone las siguientes APIs:

### Endpoints de Consulta

- `POST /api/query`: Para enviar consultas y recibir respuestas del agente experto.

### Endpoints de Reportes

- `POST /api/report/generate`: Inicia la generación de un informe en segundo plano.
- `GET /api/report/{report_id}`: Obtiene el estado de un informe en generación.
- `GET /api/report/{report_id}/download`: Descarga un informe en formato Word.
- `GET /api/report/preview/{report_id}`: Obtiene una vista previa HTML del informe.
- `POST /api/report/{report_id}/annotations`: Guarda anotaciones asociadas a un reporte.

### WebSockets

- `/ws/report`: Canal para recibir actualizaciones en tiempo real sobre la generación de informes.

## Requisitos del Sistema

### Tecnologías Principales

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Frontend**: React, Material UI
- **Base de Datos**: Supabase (PostgreSQL + extensión vectorial)
- **Servicios de IA**: OpenAI API
- **Contenedorización**: Docker

### Dependencias Principales

#### Backend:
- pydantic_ai
- openai
- supabase
- python-docx
- tiktoken
- logfire

#### Frontend:
- react
- @mui/material
- axios
- mammoth (para visualización DOCX)
- recharts (para visualizaciones)

## Instalación y Configuración

### Configuración del Backend

1. **Clonar el repositorio**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>/backend
   ```

2. **Crear entorno virtual (dentro de la carpeta backend):**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno** (crear archivo .env):
   ```
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```

5. **Iniciar el servidor**:
   ```bash
   uvicorn app.main:app --reload
   ```

### Configuración del Frontend

1. **Navegar a la carpeta del frontend**:
   ```bash
   cd ../frontend
   ```

2. **Instalar dependencias**:
   ```bash
   npm install
   ```

3. **Configurar variables de entorno** (crear archivo .env):
   ```
   VITE_API_URL=http://localhost:8000
   ```

4. **Iniciar el servidor de desarrollo**:
   ```bash
   npm run dev
   ```

### Realizar Consultas

1. Acceder a la interfaz web en `http://localhost:5173`
2. Ingresar la consulta en lenguaje natural en el campo de texto
3. El sistema procesará la consulta y mostrará la respuesta en la interfaz de chat

### Generar Informes

1. Para solicitar un informe, hacer una consulta específica como "Genera un informe sobre [tema regulatorio]"
2. El sistema iniciará la generación del informe en segundo plano
3. Cuando esté listo, se podrá visualizar y descargar desde la interfaz

## Modelo de Datos

El sistema utiliza los siguientes modelos principales:

### Base de Datos Vectorial

Tabla `documentacion_regulatoria`:
- url: String (identificador)
- chunk_number: Integer
- title: String
- summary: String
- content: String
- metadata: JSON
- embedding: Vector (1536)

### Modelos de Aplicación

- QueryRequest/QueryResponse
- ReportRequest/ReportResponse
- Annotation


### Pruebas

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

