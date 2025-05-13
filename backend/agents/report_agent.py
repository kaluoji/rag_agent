# agents/report_agent.py
from __future__ import annotations as _annotations

import os
import datetime
import logging
from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from app.core.config import settings
from utils.ppt_report import generate_ppt_report, prepare_complete_placeholders

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización del modelo
llm = settings.llm_model
model = OpenAIModel(llm, api_key=settings.openai_api_key)

# Dependencias para el agente de reportes
class ReportDeps(BaseModel):
    output_folder: str = "output/reports"  # Carpeta donde se guardarán los reportes
    openai_client: AsyncOpenAI  # Por si se requiere generar contenido adicional

    class Config:
        arbitrary_types_allowed = True

# Resultado del reporte generado
class ReportResult(BaseModel):
    file_path: str  # Ruta del archivo generado
    report_type: str = "word"  # Tipo de reporte: "word" o "ppt"
    message: str = "Reporte generado exitosamente"

# System prompt para orientar al agente en la generación del informe
report_system_prompt = """
Eres un agente experto en generación de reportes normativos sobre regulaciones. Tu tarea es crear informes estructurados que comuniquen claramente los hallazgos y recomendaciones.

Para informes en formato Word, debes incluir:
1. Portada (título, subtítulo y fecha)
2. Executive Summary
3. Alcance
4. Análisis Regulatorio / Findings
5. Conclusiones
6. Recomendaciones
7. Referencias

Utiliza la información proporcionada para detallar cada sección de forma clara y estructurada, siguiendo las normativas específicas según corresponda.
"""

# Inicialización del agente de reportes
report_agent = Agent(
    model=model,
    system_prompt=report_system_prompt,
    deps_type=ReportDeps,
    result_type=ReportResult,
    retries=2
)

def save_report_as_word(report_text: str, file_path: str) -> str:
    """
    Convierte un texto formateado en un documento de Word (.docx) y lo guarda en la ruta indicada.
    
    Args:
        report_text: Texto en formato markdown con el contenido del reporte
        file_path: Ruta donde guardar el archivo
        
    Returns:
        str: Ruta del archivo guardado
    """
    document = Document()
    
    # Separa el contenido para identificar encabezados y párrafos
    for line in report_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue  # Ignora líneas vacías
        if stripped_line.startswith("## "):
            document.add_heading(stripped_line.replace("## ", ""), level=2)
        elif stripped_line.startswith("# "):
            document.add_heading(stripped_line.replace("# ", ""), level=1)
        else:
            document.add_paragraph(stripped_line)
    
    document.save(file_path)
    return file_path

@report_agent.tool
async def generate_regulatory_report(ctx: RunContext[ReportDeps], analysis_data: str, title: str = "Informe Normativo", subtitle: str = "", date_str: str = None, format_type: str = "word") -> ReportResult:
    """
    Genera un reporte en formato Word que contenga un análisis exhaustivo de la regulación.
    
    Args:
        analysis_data: Texto con el análisis regulatorio obtenido del sistema.
        title: Título principal del reporte.
        subtitle: Subtítulo o descripción breve.
        date_str: Fecha en formato string (se usa la fecha actual si no se proporciona).
        format_type: Tipo de reporte a generar (solo "word" es soportado actualmente).
        
    Returns:
        ReportResult con la ruta del documento generado.
    """
    try:
        # Usar la fecha actual si no se especifica
        if date_str is None:
            date_str = datetime.datetime.now().strftime("%d/%m/%Y")
        
        # Crear el documento Word
        doc = Document()
        
        # --- Portada ---
        title_paragraph = doc.add_paragraph()
        title_run = title_paragraph.add_run(title)
        title_run.font.size = Pt(24)
        title_run.bold = True
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        if subtitle:
            subtitle_paragraph = doc.add_paragraph()
            subtitle_run = subtitle_paragraph.add_run(subtitle)
            subtitle_run.font.size = Pt(14)
            subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        date_paragraph = doc.add_paragraph(date_str)
        date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_page_break()
        
        # Crear el prompt para generar las secciones del informe
        prompt = f"""
Con base en el siguiente análisis sobre normativas (según la que se solicite):

{analysis_data}

Genera un informe estructurado con las siguientes secciones:
1. Executive Summary (resumen ejecutivo breve)
2. Alcance (define el alcance de este informe)
3. Análisis Regulatorio (los principales requisitos de cumplimiento basados en el análisis proporcionado)
4. Conclusiones (conclusiones importantes sobre los requisitos)
5. Recomendaciones (recomendaciones concretas para comercios)
6. Referencias (fuentes de información adicionales)

Responde con un JSON estructurado así:
{{
  "executive_summary": "texto...",
  "scope": "texto...",
  "analysis": "texto...",
  "conclusions": "texto...",
  "recommendations": "texto...",
  "references": "texto..."
}}
"""

        # Generar el contenido del informe usando directamente el cliente de OpenAI
        completion = await ctx.deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Eres un experto en la generación de informes normativos. Crea un informe estructurado, profesional y detallado."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extraer el contenido JSON
        import json
        report_content = json.loads(completion.choices[0].message.content)
        
        # --- Añadir secciones al documento ---
        # Executive Summary
        doc.add_heading("1. Executive Summary", level=1)
        doc.add_paragraph(report_content.get("executive_summary", ""))
        
        # Alcance
        doc.add_heading("2. Alcance", level=1)
        doc.add_paragraph(report_content.get("scope", ""))
        
        # Análisis Regulatorio
        doc.add_heading("3. Análisis Regulatorio", level=1)
        doc.add_paragraph(report_content.get("analysis", ""))
        
        # Conclusiones
        doc.add_heading("4. Conclusiones", level=1)
        doc.add_paragraph(report_content.get("conclusions", ""))
        
        # Recomendaciones
        doc.add_heading("5. Recomendaciones", level=1)
        doc.add_paragraph(report_content.get("recommendations", ""))
        
        # Referencias
        doc.add_heading("6. Referencias", level=1)
        doc.add_paragraph(report_content.get("references", ""))
        
        # Guardar el documento en la carpeta especificada
        output_folder = ctx.deps.output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = f"Reporte_Normativo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        file_path = os.path.join(output_folder, file_name)
        doc.save(file_path)
        
        logger.info(f"Reporte generado exitosamente en: {file_path}")
        return ReportResult(file_path=file_path, report_type="word")
    except Exception as e:
        logger.error(f"Error generando el reporte: {e}")
        raise e

@report_agent.tool
async def create_compliance_report(ctx: RunContext[ReportDeps], compliance_output: str, template: str = None) -> ReportResult:
    """
    Genera un informe de compliance utilizando el output del Agente de Compliance y lo guarda como un archivo de Word.
    
    Args:
        compliance_output: Texto con el análisis de compliance
        template: Plantilla opcional en formato markdown para estructurar el reporte
        
    Returns:
        ReportResult: Resultado con la ruta del archivo generado
    """
    # Si no se proporciona una plantilla, se utiliza una por defecto (en Markdown)
    if template is None:
        template = (
            "# Compliance Report\n\n"
            "## Executive Summary\n"
            "{executive_summary}\n\n"
            "## Findings\n"
            "{findings}\n\n"
            "## Recommendations\n"
            "{recommendations}\n\n"
            "## Conclusion\n"
            "{conclusion}\n"
        )
    
    # Crear el prompt combinando el output y la plantilla
    prompt = (
        f"Con base en el siguiente análisis de compliance:\n\n{compliance_output}\n\n"
        "Genera UN INFORME FINAL y DETALLADO que incluya las secciones: Executive Summary, Findings, Recommendations y Conclusion. "
        "DEBERÁS devolver únicamente el informe final, sin solicitar acciones adicionales, sin invocar herramientas, ni generar instrucciones para continuar. "
        "Utiliza la siguiente plantilla y reemplaza cada placeholder con la información correspondiente:\n\n"
        f"{template}\n"
    )

    # Llamada al modelo
    completion = await ctx.deps.openai_client.chat.completions.create(
        model=llm,
        temperature=0.1,
        messages=[
            {"role": "system", "content": "Eres un experto en la generación de informes normativos. Crea un informe estructurado y profesional."},
            {"role": "user", "content": prompt}
        ]
    )
    report_content = completion.choices[0].message.content
    
    # Asegurarse de que la carpeta output exista
    output_dir = ctx.deps.output_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Definir el nombre del archivo con marca de tiempo
    file_name = f"compliance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    output_file = os.path.join(output_dir, file_name)
    
    # Guardar el informe como Word
    save_report_as_word(report_text, output_file)
    
    return ReportResult(
        file_path=output_file,
        report_type="word",
        message="Informe de compliance generado exitosamente"
    )

async def process_report_query(query: str, analysis_data: str, deps: ReportDeps, format_type: str = "word") -> ReportResult:
    """
    Procesa una consulta para generar un reporte normativo.
    """
    logger.info(f"Procesando consulta para generación de reporte en formato {format_type}: {query[:100]}..." if len(query) > 100 else query)
    
    try:
        # Generar un título apropiado basado en la consulta
        completion = await deps.openai_client.chat.completions.create(
            model=llm,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Genera un título y subtítulo apropiados para un reporte normativo basado en esta consulta. Responde en formato JSON: {\"title\": \"...\", \"subtitle\": \"...\"}"},
                {"role": "user", "content": query}
            ]
        )
        
        result = completion.choices[0].message.content
        import json
        title_data = json.loads(result)
        
        title = title_data.get("title", "Informe Normativo")
        subtitle = title_data.get("subtitle", "")
        
        # Crear el documento Word
        try:
            # Importar bibliotecas necesarias
            from docx import Document
            from docx.shared import Pt, RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            import os
            import datetime
            
            # Usar la fecha actual
            date_str = datetime.datetime.now().strftime("%d/%m/%Y")
            
            # Crear el documento Word
            doc = Document()
            
            # --- Portada ---
            title_paragraph = doc.add_paragraph()
            title_run = title_paragraph.add_run(title)
            title_run.font.size = Pt(24)
            title_run.bold = True
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            if subtitle:
                subtitle_paragraph = doc.add_paragraph()
                subtitle_run = subtitle_paragraph.add_run(subtitle)
                subtitle_run.font.size = Pt(14)
                subtitle_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            date_paragraph = doc.add_paragraph(date_str)
            date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            doc.add_page_break()
            
            # Generar el contenido sección por sección para evitar límites
            
            # 1. Executive Summary
            exec_summary_prompt = f"Basado en el siguiente análisis normativo: \n\n{analysis_data[:2000]}...\n\nGenera un resumen ejecutivo conciso (5-10 oraciones) que destaque los puntos clave del informe sobre la normativa analizada."
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en informes normativos. Genera un resumen ejecutivo profesional y conciso."},
                    {"role": "user", "content": exec_summary_prompt}
                ]
            )
            
            exec_summary = completion.choices[0].message.content
            
            # 2. Alcance
            scope_prompt = f"Basado en esta consulta: '{query}' y este análisis normativo: \n\n{analysis_data[:1000]}...\n\nGenera una sección de 'Alcance' que defina claramente el propósito y límites del informe."
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en informes normativos. Define claramente el alcance del informe."},
                    {"role": "user", "content": scope_prompt}
                ]
            )
            
            scope = completion.choices[0].message.content
            
            # 3. Análisis Regulatorio (parte más importante)
            analysis_prompt = f"""Basado en el siguiente análisis normativo: 

{analysis_data}

Genera un análisis regulatorio completo y detallado. Incluye:
1. Los principales requisitos normativos identificados
2. Las obligaciones para las entidades reguladas
3. Los procesos de cumplimiento necesarios
4. Los riesgos y consecuencias del incumplimiento

Estructura la información en subsecciones con títulos claros. Incluye datos específicos, cifras y referencias directas a la normativa cuando sea posible.
"""
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                max_tokens=2048,  # Permitir respuestas más largas
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en análisis normativo. Elabora análisis detallados, específicos y bien estructurados."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            regulatory_analysis = completion.choices[0].message.content
            
            # 4. Conclusiones
            conclusions_prompt = f"Basado en este análisis normativo: \n\n{regulatory_analysis}\n\nGenera conclusiones clave sobre la normativa analizada, incluyendo aspectos críticos a considerar y posibles implicaciones."
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en informes normativos. Elabora conclusiones fundamentadas basadas en el análisis previo."},
                    {"role": "user", "content": conclusions_prompt}
                ]
            )
            
            conclusions = completion.choices[0].message.content
            
            # 5. Recomendaciones
            recommendations_prompt = f"Basado en este análisis normativo y conclusiones: \n\n{conclusions}\n\nGenera recomendaciones prácticas y accionables para el cumplimiento de la normativa analizada."
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en informes normativos. Proporciona recomendaciones específicas y prácticas."},
                    {"role": "user", "content": recommendations_prompt}
                ]
            )
            
            recommendations = completion.choices[0].message.content
            
            # 6. Referencias
            references_prompt = f"Basado en este análisis normativo: \n\n{analysis_data[:1000]}...\n\nIdentifica las principales fuentes, documentos oficiales o normativas mencionadas que deberían incluirse en una sección de referencias."
            
            completion = await deps.openai_client.chat.completions.create(
                model=llm,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Eres un experto en informes normativos. Identifica las referencias relevantes."},
                    {"role": "user", "content": references_prompt}
                ]
            )
            
            references = completion.choices[0].message.content
            
            # --- Añadir secciones al documento ---
            def add_formatted_section(doc, title, content):
                """Añade una sección formateada al documento."""
                doc.add_heading(title, level=1)
                
                # Detectar y procesar posibles subsecciones
                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Detectar posibles títulos de subsección
                    if (line.startswith('#') or 
                        line.endswith(':') and len(line) < 100 and i+1 < len(lines) and lines[i+1].strip()):
                        # Es un título de subsección
                        if line.startswith('#'):
                            line = line.lstrip('#').strip()
                        if line.endswith(':'):
                            line = line[:-1].strip()
                            
                        doc.add_heading(line, level=2)
                        i += 1
                    elif line.startswith('-') or line.startswith('*') or (line.startswith(str(i+1)+'.') and len(line) < 100):
                        # Es un elemento de lista
                        bullet_text = line[line.find(' ')+1:].strip()
                        p = doc.add_paragraph(style='List Bullet')
                        p.add_run(bullet_text)
                        i += 1
                    elif line.strip():
                        # Es un párrafo normal
                        paragraph_text = line
                        
                        # Acumular líneas para párrafos multilínea
                        j = i + 1
                        while j < len(lines) and lines[j].strip() and not (
                            lines[j].startswith('#') or 
                            lines[j].endswith(':') and len(lines[j]) < 100 or
                            lines[j].startswith('-') or 
                            lines[j].startswith('*') or
                            lines[j].startswith(str(j+1)+'.')
                        ):
                            paragraph_text += ' ' + lines[j].strip()
                            j += 1
                        
                        doc.add_paragraph(paragraph_text)
                        i = j
                    else:
                        # Línea vacía
                        i += 1
            
            # Añadir cada sección con formato mejorado
            add_formatted_section(doc, "1. Executive Summary", exec_summary)
            add_formatted_section(doc, "2. Alcance", scope)
            add_formatted_section(doc, "3. Análisis Regulatorio", regulatory_analysis)
            add_formatted_section(doc, "4. Conclusiones", conclusions)
            add_formatted_section(doc, "5. Recomendaciones", recommendations)
            add_formatted_section(doc, "6. Referencias", references)
            
            # Guardar el documento en la carpeta especificada
            output_folder = deps.output_folder
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            file_name = f"Reporte_Normativo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            file_path = os.path.join(output_folder, file_name)
            doc.save(file_path)
            
            logger.info(f"Reporte generado exitosamente en: {file_path}")
            return ReportResult(file_path=file_path, report_type="word", message="Reporte generado exitosamente")
        
        except Exception as e:
            logger.error(f"Error generando el documento Word: {e}")
            raise e
        
    except Exception as e:
        logger.error(f"Error procesando consulta para reporte: {e}")
        # En caso de error, crear un resultado con mensaje de error
        return ReportResult(
            file_path="",
            message=f"Error generando el reporte: {str(e)}"
        )