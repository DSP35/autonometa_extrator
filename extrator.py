# Extrator Autonometa
# Desenvolvido por David Parede

import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO
import json
import os
import xml.etree.ElementTree as ET 
import re
from typing import Optional

import pandas as pd
import plotly.express as px

import cv2 
import numpy as np

from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import ValidationError

# --- CONFIGURAÇÕES GERAIS ---
TESSERACT_PATH = '/usr/bin/tesseract'
if 'TESSERACT_PATH' in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_PATH']
elif os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    # Apenas em ambientes locais se for necessário
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
    pass

st.set_page_config(
    page_title="Extrator Autonometa",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- MODELOS PYDANTIC (Schema de Saída) ---

def formatar_moeda_imp(valor):
    """Formata valor float para exibição como moeda brasileira."""
    try:
        if valor is None:
            return "R$ 0,00"
        return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "R$ 0,00"

class ParteFiscal(BaseModel):
    cnpj_cpf: str = Field(description="CNPJ ou CPF da parte fiscal (apenas dígitos).")
    
    # REINTRODUZIDO: Aceita 'nome_raza' do LLM como input para o campo 'nome_razao'
    nome_razao: str = Field(
        description="Nome ou Razão Social completa.",
        validation_alias='nome_raza' 
    )
    
    endereco_completo: str = Field(description="Endereço completo (Rua, Número, Bairro, Cidade, Estado).")
    inscricao_estadual: str = Field(description="Inscrição Estadual, se disponível.")

class ItemFiscal(BaseModel):
    descricao: str = Field(description="Descrição completa do item ou serviço.")
    quantidade: float = Field(description="Quantidade numérica do item.")
    valor_unitario: float = Field(description="Valor unitário do item (float).")
    valor_total: float = Field(description="Valor total do item (float).")
    codigo_cfop: str = Field(description="CFOP do item (apenas dígitos).")
    cst_csosn: str = Field(description="CST ou CSOSN do item (apenas dígitos).")
    valor_aprox_tributos: float = Field(description="Valor aproximado dos tributos do item (float).")

class TotaisImpostos(BaseModel):
    base_calculo_icms: float = Field(description="Base de Cálculo do ICMS (float).")
    valor_total_icms: float = Field(description="Valor Total do ICMS (float).")
    valor_total_ipi: float = Field(description="Valor Total do IPI (float).")
    valor_total_pis: float = Field(description="Valor Total do PIS (float).")
    valor_total_cofins: float = Field(description="Valor Total do COFINS (float).")
    valor_outras_despesas: float = Field(description="Outras Despesas Acessórias (float).")
    valor_aprox_tributos: float = Field(description="Valor total aproximado de tributos (Lei da Transparência) (float).")

class NotaFiscal(BaseModel):
    chave_acesso: str = Field(description="Chave de Acesso da NF-e (44 dígitos).")
    modelo_documento: str = Field(description="Modelo do documento fiscal (ex: NF-e, NFS-e).")
    data_emissao: str = Field(description="Data de emissão (formato AAAA-MM-DD).")
    valor_total_nota: float = Field(description="Valor total final da nota (float).")
    natureza_operacao: str = Field(description="Natureza da Operação (ex: Venda de Mercadoria).")
    
    emitente: ParteFiscal = Field(description="Dados do emitente/remetente.")
    destinatario: ParteFiscal = Field(description="Dados do destinatário.")
    
    totais_impostos: TotaisImpostos = Field(description="Valores totais e impostos da nota.")
    itens: list[ItemFiscal] = Field(description="Lista de produtos ou serviços (itens) na nota.")
    
# --- FUNÇÕES DE PRÉ-PROCESSAMENTO (PONTO 1 e 2) ---

def get_image_brightness(image_np):
    """Calcula o brilho médio da imagem (escala de cinza)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def preprocess_image_for_ocr(image_pil: Image.Image) -> np.ndarray:
    """
    Aplica pré-processamento (OpenCV) para aumentar a robustez do OCR.
    Passos: Binarização adaptativa e Remoção de Ruído.
    """
    
    # 1. Converte PIL Image para array numpy (BGR)
    image_np = np.array(image_pil.convert('RGB'))
    image_np = image_np[:, :, ::-1].copy()
    
    # 2. Converte para Escala de Cinza
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 3. Alerta básico de qualidade (Brilho)
    brightness = get_image_brightness(image_np)
    if brightness < 80:
        st.sidebar.warning(f"⚠️ Nota: Imagem escura (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")
    elif brightness > 220:
         st.sidebar.warning(f"⚠️ Nota: Imagem muito clara (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")
         
    # 4. Suavização (Remoção de Ruído)
    denoised = cv2.medianBlur(gray, 3) 
    
    # 5. Binarização Adaptativa
    processed_image = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11,
        2
    )
    
    return processed_image

def extract_text_from_file(uploaded_file):
    """
    Processa o arquivo carregado (JPG/PNG ou PDF) e retorna o texto extraído
    usando Tesseract OCR, com pré-processamento do OpenCV.
    Se for PDF multipágina, concatena o texto de todas as páginas.
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)
    
    tesseract_config = '--psm 4' 
    
    full_text_list = []
    images_to_process = []
    img_to_display = None
    
    # 1. Se for PDF (Múltiplas páginas)
    if "pdf" in file_type:
        try:
            images_to_process = convert_from_bytes(uploaded_file.read())
            
            if not images_to_process:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."
            
            img_to_display = images_to_process[0]
            
        except Exception as e:
            return f"ERRO_PDF: Verifique se 'poppler-utils' está instalado via packages.txt. Detalhes: {e}"

    # 2. Se for Imagem (Página única)
    elif "image" in file_type:
        try:
            img_to_display = Image.open(uploaded_file)
            images_to_process.append(img_to_display)
        except Exception as e:
            return f"ERRO_IMAGEM: Falha na abertura da imagem. Detalhes: {e}"
            
    else:
        return "ERRO_TIPO_INVALIDO: Tipo de arquivo não suportado."
    
    # --- PROCESSAMENTO ITERATIVO (OCR + PRÉ-PROCESSAMENTO) ---
    if images_to_process:
        try:
            for i, image_pil in enumerate(images_to_process):
                # Pré-processamento (OpenCV)
                img_for_ocr = preprocess_image_for_ocr(image_pil)
                
                # Executa o OCR
                text = pytesseract.image_to_string(img_for_ocr, lang='por', config=tesseract_config)
                
                full_text_list.append(f"\n--- INÍCIO PÁGINA {i+1} ---\n\n" + text)
            
            if img_to_display is not None:
                st.session_state["image_to_display"] = img_to_display 
            
            return "\n".join(full_text_list)

        except pytesseract.TesseractNotFoundError:
            return "ERRO_IMAGEM: O Tesseract não está instalado corretamente via packages.txt."
        except Exception as e:
            return f"ERRO_PROCESSAMENTO: Falha no OCR ou pré-processamento. Detalhes: {e}"
            
    return "ERRO_FALHA_GERAL: Falha desconhecida na extração de texto."

# --- FUNÇÕES DE AUDITORIA E ENRIQUECIMENTO (PONTO 3) ---

def check_for_missing_data(data_dict: dict) -> list[str]:
    """Verifica se campos críticos estão faltando ou zerados."""
    warnings = []
    
    if not data_dict.get('chave_acesso'):
        warnings.append("- Chave de Acesso: O campo chave_acesso está vazio.")
    if data_dict.get('valor_total_nota', 0.0) <= 0.0:
        warnings.append(f"- Valor Total da Nota: Valor zerado ou ausente ({data_dict.get('valor_total_nota')}).")
    if not data_dict.get('emitente', {}).get('nome_razao'):
        warnings.append("- Emitente: Nome/Razão Social ausente.")
    if not data_dict.get('destinatario', {}).get('nome_razao'):
        warnings.append("- Destinatário: Nome/Razão Social ausente.")
    if not data_dict.get('itens'):
        warnings.append("- Itens: Nenhum item de produto/serviço foi extraído.")
    
    return warnings

def enrich_and_validate_extraction(data_dict: dict, ocr_text: str) -> tuple[dict, list]:
    """
    1. Executa fallback heurístico (Regex) para CFOP/CST/CSOSN em itens.
    2. Executa pós-validação comparando o total de itens com o total da nota.
    Retorna o dicionário enriquecido e uma lista de mensagens para exibição.
    """
    
    enriched_data = data_dict.copy()
    itens_processados = []
    total_itens_calculado = 0.0
    messages = []
    
    cfop_pattern = re.compile(r'\b(\d{4})\b')
    cst_pattern = re.compile(r'\b(0\d{2}|[1-9]\d{1,2})\b')
    
    # 1. Fallback Heurístico (Regex) para Itens
    if ocr_text:
        messages.append(("info", "Iniciando enriquecimento heurístico para códigos fiscais (CFOP, CST/CSOSN)."))
        
        for item in enriched_data.get('itens', []):
            item_desc_lower = item['descricao'].lower()
            
            try:
                item['valor_total'] = float(item['valor_total'])
            except (TypeError, ValueError):
                 item['valor_total'] = 0.0
                 
            total_itens_calculado += item['valor_total']

            
            # Fallback para CFOP
            if not item.get('codigo_cfop') or len(str(item['codigo_cfop'])) != 4:
                match = cfop_pattern.search(item_desc_lower)
                if match:
                    item['codigo_cfop'] = match.group(1)
                    messages.append(("success", f"CFOP do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['codigo_cfop']}**"))


            # Fallback para CST/CSOSN
            if not item.get('cst_csosn') or len(str(item['cst_csosn'])) < 2:
                match = cst_pattern.search(item_desc_lower)
                if match:
                    item['cst_csosn'] = match.group(1)
                    messages.append(("success", f"CST/CSOSN do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['cst_csosn']}**"))

            itens_processados.append(item)
    
    enriched_data['itens'] = itens_processados
    
    
    # 2. Pós-validação de Totais
    
    messages.append(("info", "Iniciando pós-validação de consistência de totais."))
    
    valor_total_nota = enriched_data.get('valor_total_nota', 0.0)
    tolerance = 0.01 
    
    soma_itens_formatada = formatar_moeda_imp(total_itens_calculado)
    total_nf_formatado = formatar_moeda_imp(valor_total_nota)

    if abs(total_itens_calculado - valor_total_nota) <= tolerance:
        messages.append(("success", f"👍 Consistência Aprovada! O somatório dos itens é consistente com o Valor Total da Nota. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado}"))
    else:
        messages.append(("error", f"🚨 ALERTA DE INCONSISTÊNCIA! O somatório dos itens extraídos é diferente do Valor Total da Nota extraído. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado} | Recomendação: Verifique a qualidade do OCR ou edite os valores manualmente."))
        
    return enriched_data, messages

# --- FUNÇÃO DE EXTRAÇÃO XML ---

def get_xml_data(uploaded_file):
    """
    Extrai dados estruturados de um arquivo XML (NF-e padrão)
    e retorna no formato de dicionário compatível com o Pydantic.
    """
    
    uploaded_file.seek(0)
    xml_data = uploaded_file.read()
    
    # Tenta usar ElementTree
    try:
        root = ET.fromstring(xml_data)
        
        # Define o namespace padrão para evitar prefixos
        # Isso é crucial para NF-e
        namespace = {'nfe': root.tag.split('}')[0].strip('{')}
        if not namespace['nfe']:
             namespace['nfe'] = 'http://www.portalfiscal.inf.br/nfe' 
        
        inf_nfe = root.find('.//nfe:infNFe', namespace)
        ide = root.find('.//nfe:ide', namespace)
        emit = root.find('.//nfe:emit', namespace)
        dest = root.find('.//nfe:dest', namespace)
        total = root.find('.//nfe:total', namespace)
        icms_tot = root.find('.//nfe:ICMSTot', namespace)
        
        # Extração de Itens
        itens_xml = []
        for det in root.findall('.//nfe:det', namespace):
            prod = det.find('.//nfe:prod', namespace)
            imposto = det.find('.//nfe:imposto', namespace)
            
            # Tenta encontrar ICMS
            icms_node = imposto.find('.//nfe:ICMS', namespace) if imposto is not None else None
            cst_csosn = '00' # Default
            if icms_node is not None:
                # Tenta ICMSXX, onde XX é qualquer número (ex: ICMS00, ICMS40)
                icms_sub_node = icms_node.find('./*', namespace)
                if icms_sub_node is not None:
                    cst_csosn = icms_sub_node.find('.//nfe:CST', namespace).text if icms_sub_node.find('.//nfe:CST', namespace) is not None else cst_csosn
                    if cst_csosn == '00' or cst_csosn is None:
                         cst_csosn = icms_sub_node.find('.//nfe:CSOSN', namespace).text if icms_sub_node.find('.//nfe:CSOSN', namespace) is not None else '00'


            item = {
                "descricao": prod.find('.//nfe:xProd', namespace).text if prod is not None else "",
                "quantidade": float(prod.find('.//nfe:qCom', namespace).text) if prod is not None else 0.0,
                "valor_unitario": float(prod.find('.//nfe:vUnCom', namespace).text) if prod is not None else 0.0,
                "valor_total": float(prod.find('.//nfe:vProd', namespace).text) if prod is not None else 0.0,
                "codigo_cfop": prod.find('.//nfe:CFOP', namespace).text if prod is not None else "",
                "cst_csosn": cst_csosn,
                # Não há um campo direto para V.Aprox. Tributos por item no XML padrão sem cálculo
                "valor_aprox_tributos": 0.0
            }
            itens_xml.append(item)

        
        # Extração de Dados Principais
        data_dict = {
            "chave_acesso": inf_nfe.attrib.get('Id', '').replace('NFe', ''),
            "modelo_documento": ide.find('.//nfe:mod', namespace).text if ide is not None else "",
            "data_emissao": ide.find('.//nfe:dhEmi', namespace).text[:10].replace('-', '/') if ide is not None else "", # Simplifica data
            "valor_total_nota": float(icms_tot.find('.//nfe:vNF', namespace).text) if icms_tot is not None else 0.0,
            "natureza_operacao": ide.find('.//nfe:natOp', namespace).text if ide is not None else "",
            
            "emitente": {
                "cnpj_cpf": emit.find('.//nfe:CNPJ', namespace).text if emit.find('.//nfe:CNPJ', namespace) is not None else emit.find('.//nfe:CPF', namespace).text,
                "nome_razao": emit.find('.//nfe:xNome', namespace).text,
                "endereco_completo": f"{emit.find('.//nfe:xLgr', namespace).text}, {emit.find('.//nfe:nro', namespace).text} - {emit.find('.//nfe:xBairro', namespace).text}, {emit.find('.//nfe:xMun', namespace).text} - {emit.find('.//nfe:UF', namespace).text}",
                "inscricao_estadual": emit.find('.//nfe:IE', namespace).text,
            },
            
            "destinatario": {
                "cnpj_cpf": dest.find('.//nfe:CNPJ', namespace).text if dest.find('.//nfe:CNPJ', namespace) is not None else dest.find('.//nfe:CPF', namespace).text,
                "nome_razao": dest.find('.//nfe:xNome', namespace).text,
                "endereco_completo": f"{dest.find('.//nfe:xLgr', namespace).text}, {dest.find('.//nfe:nro', namespace).text} - {dest.find('.//nfe:xBairro', namespace).text}, {dest.find('.//nfe:xMun', namespace).text} - {dest.find('.//nfe:UF', namespace).text}",
                "inscricao_estadual": dest.find('.//nfe:IE', namespace).text if dest.find('.//nfe:IE', namespace) is not None else ""
            },
            
            "totais_impostos": {
                "base_calculo_icms": float(icms_tot.find('.//nfe:vBC', namespace).text) if icms_tot is not None else 0.0,
                "valor_total_icms": float(icms_tot.find('.//nfe:vICMS', namespace).text) if icms_tot is not None else 0.0,
                "valor_total_ipi": float(icms_tot.find('.//nfe:vIPI', namespace).text) if icms_tot is not None else 0.0,
                "valor_total_pis": float(icms_tot.find('.//nfe:vPIS', namespace).text) if icms_tot is not None else 0.0,
                "valor_total_cofins": float(icms_tot.find('.//nfe:vCOFINS', namespace).text) if icms_tot is not None else 0.0,
                "valor_outras_despesas": float(icms_tot.find('.//nfe:vOutr', namespace).text) if icms_tot is not None else 0.0,
                # Valor aproximado de tributos é extraído de infAdic
                "valor_aprox_tributos": 0.0 
            },
            "itens": itens_xml
        }
        
        # Pós-processamento de data (de AAAA-MM-DDTHH:MM:SS para AAAA-MM-DD)
        if data_dict['data_emissao'] and 'T' in data_dict['data_emissao']:
            data_dict['data_emissao'] = data_dict['data_emissao'].split('T')[0]
        
        return data_dict

    except Exception as e:
        return {"error": f"Erro ao processar o arquivo XML. O arquivo pode estar malformado ou não seguir o schema NF-e. Detalhes: {e}"}


# --- FUNÇÃO DE EXIBIÇÃO DE RESULTADOS (DASHBOARD - PONTO 5) ---

def display_extraction_results(data_dict: dict, source: str, ocr_text: Optional[str] = None):
    """Exibe os resultados estruturados na tela principal, independentemente da fonte (XML ou LLM), e o Dashboard."""
    
    st.header(f"✅ Resultado da Extração Estruturada ({source})")
    
    # 1. Pós-validação (PONTO 3: Coleta e renderiza as mensagens)
    if source == "LLM/OCR" and ocr_text:
        data_dict, audit_messages = enrich_and_validate_extraction(data_dict, ocr_text) 
        
        st.markdown("---")
        st.subheader("🛠️ Enriquecimento e Auditoria Pós-Extração")
        
        for msg_type, msg_text in audit_messages:
            if msg_type == "info":
                st.info(msg_text)
            elif msg_type == "success":
                st.success(msg_text, icon="✔")
            elif msg_type == "error":
                st.error(msg_text, icon="❌")
            else:
                 st.markdown(msg_text)

        st.markdown("---")
        
    # 2. Checagem de Qualidade (Ponto 3)
    quality_warnings = check_for_missing_data(data_dict)
    
    if quality_warnings:
        st.warning("⚠️ Atenção: Diversas informações críticas estão faltando ou ilegíveis na nota fiscal. Isso geralmente ocorre devido à má qualidade da digitalização.")
        with st.expander("Clique para ver os campos faltantes ou zerados"):
            for warning in quality_warnings:
                st.markdown(warning)

    
    # --- 3. DASHBOARD: KPIs e Indicadores (Ponto 5) ---
    st.subheader("📊 Resumo Fiscal (KPIs)")
    
    impostos_data = data_dict.get('totais_impostos', {})
    valor_total = data_dict.get('valor_total_nota', 0.0)
    total_itens = len(data_dict.get('itens', []))
    total_tributos = impostos_data.get('valor_aprox_tributos', 0.0)
    total_icms = impostos_data.get('valor_total_icms', 0.0)
    total_ipi = impostos_data.get('valor_total_ipi', 0.0)
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    kpi1.metric("Valor Total da NF", formatar_moeda_imp(valor_total).replace("R$ ", ""))
    kpi2.metric("V. Aprox. Tributos", formatar_moeda_imp(total_tributos).replace("R$ ", ""))
    kpi3.metric("Total ICMS", formatar_moeda_imp(total_icms).replace("R$ ", ""))
    kpi4.metric("Nº de Itens", total_itens)
    kpi5.metric("Inconsistências", len(quality_warnings), delta="Críticas encontradas", delta_color="inverse")
    
    st.markdown("---")
    
    
    # --- 4. DETALHES GERAIS DA NOTA ---
    st.subheader("Informações Principais")
    
    col_data, col_valor, col_modelo, col_natureza = st.columns(4)
    
    col_data.metric("Data de Emissão", data_dict['data_emissao'])
    col_valor.metric("Valor Total da Nota", formatar_moeda_imp(data_dict.get('valor_total_nota', 0.0)).replace("R$ ", "")) 
    col_modelo.metric("Modelo Fiscal", data_dict['modelo_documento'])
    col_natureza.metric("Natureza da Operação", data_dict['natureza_operacao'])


    st.markdown("#### 🔑 **Chave de Acesso da NF-e**")
    st.code(data_dict['chave_acesso'], language="text")

    st.markdown("---")
    
    
    # 5. Detalhes do Emitente e Destinatário
    col_emitente, col_destinatario = st.columns(2)
    
    with col_emitente.expander("🏢 Detalhes do Emitente", expanded=False):
        emitente_data = data_dict.get('emitente', {})
        st.json(emitente_data)

    with col_destinatario.expander("👤 Detalhes do Destinatário", expanded=False):
        destinatario_data = data_dict.get('destinatario', {})
        st.json(destinatario_data)


    # 6. Tabela de Itens e Gráfico (Ponto 5)
    st.subheader("🛒 Itens da Nota Fiscal")
    
    itens_list = data_dict.get('itens', [])
    
    if itens_list:
        
        df_itens = pd.DataFrame(itens_list)
        
        for col in ['quantidade', 'valor_unitario', 'valor_total', 'valor_aprox_tributos']:
            df_itens[col] = pd.to_numeric(df_itens[col], errors='coerce').fillna(0.0)

        st.dataframe(
            df_itens,
            column_order=["descricao", "quantidade", "valor_unitario", "valor_total", "codigo_cfop", "cst_csosn", "valor_aprox_tributos"],
            column_config={
                "descricao": st.column_config.Column("Descrição do Item", width="large"),
                "quantidade": st.column_config.NumberColumn("Qtde"),
                "valor_unitario": st.column_config.NumberColumn("Valor Unit.", format="R$ %.2f"),
                "valor_total": st.column_config.NumberColumn("Valor Total", format="R$ %.2f"),
                "codigo_cfop": st.column_config.Column("CFOP"),
                "cst_csosn": st.column_config.Column("CST/CSOSN"),
                "valor_aprox_tributos": st.column_config.NumberColumn("V. Aprox. Tributos", format="R$ %.2f")
            },
            hide_index=True,
            width='stretch'
        )
        
        st.markdown("### 📈 Distribuição de Valor por CFOP")
        df_cfop = df_itens.groupby('codigo_cfop', dropna=False)['valor_total'].sum().reset_index()
        df_cfop.columns = ['CFOP', 'Valor Total']
        
        fig = px.bar(
            df_cfop, 
            x='CFOP', 
            y='Valor Total', 
            text='Valor Total',
            labels={'Valor Total': 'Valor Total (R$)', 'CFOP': 'Código Fiscal de Operações'},
            color='CFOP',
            title='Valor de Produtos/Serviços agrupado por CFOP'
        )
        fig.update_traces(texttemplate='R$ %{y:,.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        
    else:
        st.warning("Nenhum item ou serviço foi encontrado na nota fiscal.")


    # 7. Exibição dos Totais de Impostos
    st.markdown("---")
    st.subheader("💰 Totais de Impostos e Despesas")

    total_tributos_calculado = df_itens['valor_aprox_tributos'].sum() if 'df_itens' in locals() else 0.0
    total_tributos_extraido_direto = impostos_data.get('valor_aprox_tributos', 0.0)

    if total_tributos_calculado > 0.0:
        total_final_tributos = total_tributos_calculado
        fonte_tributos = " (Calculado dos Itens)"
    elif total_tributos_extraido_direto > 0.0:
        total_final_tributos = total_tributos_extraido_direto
        fonte_tributos = " (Extraído dos Dados Adicionais)"
    else:
        total_final_tributos = 0.0
        fonte_tributos = ""


    col_icms, col_ipi, col_pis, col_cofins, col_outras, col_aprox = st.columns(6)

    col_icms.metric("Base ICMS", formatar_moeda_imp(impostos_data.get('base_calculo_icms')))
    col_icms.metric("Total ICMS", formatar_moeda_imp(impostos_data.get('valor_total_icms')))

    col_ipi.metric("Total IPI", formatar_moeda_imp(impostos_data.get('valor_total_ipi')))

    col_pis.metric("Total PIS", formatar_moeda_imp(impostos_data.get('valor_total_pis')))
    col_cofins.metric("Total COFINS", formatar_moeda_imp(impostos_data.get('valor_total_cofins')))

    col_outras.metric("Outras Despesas", formatar_moeda_imp(impostos_data.get('valor_outras_despesas')))

    col_aprox.metric(f"Total V. Aprox. Tributos{fonte_tributos}", formatar_moeda_imp(total_final_tributos))
    
    
    # 8. Edição Manual Assistida
    icms_zerado = impostos_data.get('valor_total_icms', 0.0) <= 0.0
    ipi_zerado = impostos_data.get('valor_total_ipi', 0.0) <= 0.0
    
    if icms_zerado or ipi_zerado:
        st.markdown("---")
        st.subheader("✍️ Edição Manual de Impostos")
        st.info("O Agente LLM não conseguiu extrair os valores detalhados. Se a nota contém esses valores, insira-os abaixo.")
        
        icms_val = str(impostos_data.get('valor_total_icms', 0.0))
        ipi_val = str(impostos_data.get('valor_total_ipi', 0.0))
        pis_val = str(impostos_data.get('valor_total_pis', 0.0))
        cofins_val = str(impostos_data.get('valor_total_cofins', 0.0))

        col_edit_icms, col_edit_ipi, col_edit_pis, col_edit_cofins = st.columns(4)
        
        key_suffix = source.lower().replace("/", "_")
        
        icms_manual = col_edit_icms.text_input("ICMS", value=icms_val, key=f"manual_icms_{key_suffix}")
        ipi_manual = col_edit_ipi.text_input("IPI", value=ipi_val, key=f"manual_ipi_{key_suffix}")
        pis_manual = col_edit_pis.text_input("PIS", value=pis_val, key=f"manual_pis_{key_suffix}")
        cofins_manual = col_edit_cofins.text_input("COFINS", value=cofins_val, key=f"manual_cofins_{key_suffix}")
        
        try:
            data_dict['totais_impostos']['valor_total_icms'] = float(icms_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_ipi'] = float(ipi_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_pis'] = float(pis_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_cofins'] = float(cofins_manual.replace(",", "."))
            st.success("Valores de impostos atualizados para o JSON de download.")
            
        except ValueError:
            st.error("Por favor, insira apenas números válidos nos campos de edição.")


    # --- 9. Botões de Download (JSON e CSV) ---
    st.markdown("---")
    col_json_btn, col_csv_btn = st.columns(2)
    
    try:
        nome_curto = data_dict['emitente']['nome_razao'].split(' ')[0]
        data_emissao_nome = data_dict['data_emissao']
    except (KeyError, IndexError, TypeError):
        nome_curto = "extraida"
        data_emissao_nome = "data_desconhecida"

    json_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
    col_json_btn.download_button(
        label="⬇️ Baixar JSON COMPLETO da Extração",
        data=json_data,
        file_name=f"nf_{data_emissao_nome}_{nome_curto}.json",
        mime="application/json",
        use_container_width=True
    )
    
    if 'df_itens' in locals() and not df_itens.empty:
        csv_data = df_itens.to_csv(index=False).encode('utf-8')
        col_csv_btn.download_button(
            label="⬇️ Baixar Itens em CSV",
            data=csv_data,
            file_name=f"itens_{data_emissao_nome}_{nome_curto}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with st.expander("Ver JSON Bruto Completo (DEBUG)", expanded=False):
         st.json(data_dict)


# --- CONFIGURAÇÃO DO LLM ---

system_prompt = (
    "Você é um Agente de Extração Fiscal especializado em Notas Fiscais Eletrônicas (NF-e) e DANFE."
    "Sua função é ler o texto bruto (OCR) de documentos fiscais e extrair os dados em formato JSON, "
    "obedecendo rigorosamente o schema Pydantic fornecido."
    "Siga estas regras estritas:"
    "1. **Extração de Texto Bruto:** Se um campo estiver faltando ou for ilegível no texto OCR, preencha-o com uma string vazia (''), mas *nunca* invente dados."
    "2. **Valores Numéricos (CRÍTICO - FORMATO BRASILEIRO):** Converta todos os valores monetários e quantias (que usam ponto como milhar e vírgula como decimal, ex: 1.234,56) para o formato `float` americano (ponto como separador decimal, sem separador de milhar, ex: 1234.56). "
    "   - **Atenção:** Remova o separador de milhar (ponto ou espaço) e substitua a vírgula (,) pelo ponto (.)."
    "3. **Datas:** Converta todas as datas para o formato estrito 'AAAA-MM-DD'."
    "4. **Chave de Acesso:** A chave deve ser uma string de 44 dígitos (apenas números)."
    "5. **Tabelas de Itens:** Preste **MÁXIMA ATENÇÃO** à leitura correta das colunas. O campo `valor_total` deve ser o **Valor Total do Item/Produto**, e **NÃO** o Valor de ICMS ou outro imposto."
    "6. **Saída:** O resultado final deve ser **SOMENTE** o JSON, sem qualquer texto explicativo ou markdown adicional."
)

parser = PydanticOutputParser(pydantic_object=NotaFiscal)

prompt = PromptTemplate(
    template="Responda ao pedido do usuário.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- LÓGICA PRINCIPAL DO APP ---

st.title("Análise e Extração Estruturada de Dados 🧠")

uploaded_file = st.file_uploader(
    "📥 Escolha um arquivo (XML, PDF, PNG, JPG) para análise", 
    type=["xml", "pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    
    file_type = uploaded_file.type
    
    with st.spinner(f"Processando arquivo ({uploaded_file.name})..."):
        
        # --- FLUXO 1: XML (Prioridade Máxima) ---
        if "xml" in file_type:
            data_dict = get_xml_data(uploaded_file)
            
            if "error" in data_dict:
                st.error(data_dict["error"])
            else:
                try:
                    # Valida o XML extraído contra o Pydantic
                    NotaFiscal(**data_dict) 
                    display_extraction_results(data_dict, source="XML")
                except ValidationError as ve:
                    st.error(f"Erro de Validação Pydantic ao ler XML: {ve}")
                    st.info("O XML foi processado, mas falhou na validação do esquema Pydantic. Use o JSON Bruto para debug.")
                    # Continua a exibição para debug
                    display_extraction_results(data_dict, source="XML")

        # --- FLUXO 2: OCR/LLM (PDF/Imagem) ---
        else:
            
            # 1. Extração de texto bruto (OCR)
            text_to_analyze = extract_text_from_file(uploaded_file)
            
            if text_to_analyze.startswith("ERRO_"):
                 st.error(f"Erro na extração de texto (OCR): {text_to_analyze}")
                 st.markdown("Verifique se as dependências (poppler-utils, tesseract) estão instaladas.")
            else:
                # 2. Miniatura da Imagem na Sidebar
                if "image_to_display" in st.session_state:
                    st.sidebar.success("Imagem carregada e OCR inicial concluído.")
                    with st.sidebar.expander("🔎 Visualizar Nota Fiscal"):
                        st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", width='stretch')

                
                # 3. Execução do LLM
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0,
                        system_instruction=system_prompt,
                        # stream=True # Descomente se quiser ver a resposta em streaming
                    )
                    
                    # Constrói a query com o texto do OCR
                    final_prompt = prompt.format_prompt(query=text_to_analyze)
                    
                    response = llm.invoke(final_prompt)
                    extracted_data = parser.parse(response.content)
                    
                    data_dict = extracted_data.model_dump()
                        
                    # 4. CHAMA A FUNÇÃO DE DISPLAY APÓS SUCESSO DO LLM E ENRIQUECIMENTO
                    display_extraction_results(data_dict, source="LLM/OCR", ocr_text=text_to_analyze)
                    
                except ValidationError as ve:
                    st.error(f"Houve um erro durante a interpretação pelo Gemini. Detalhes: {ve}")
                    st.warning("O Agente LLM pode ter falhado ao extrair a estrutura JSON a partir do texto OCR.")
                    with st.expander("Ver Texto OCR Bruto"):
                        st.code(text_to_analyze, language="text")
                except Exception as e:
                    st.error(f"Ocorreu um erro inesperado: {e}")
