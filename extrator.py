# Extrator Autonometa
# Desenvolvido por David Parede

import streamlit as st
from PIL import Image
import pytesseract         # OCR
from pdf2image import convert_from_bytes
from io import BytesIO
import json                
import os
import xml.etree.ElementTree as ET # NOVO: Import para XML
import cv2 
import numpy as np
import re
import pandas as pd
import plotly.express as px

# --- Imports LangChain e Pydantic ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate        
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel, Field, ValidationError
from typing import Optional 

# --- 1. Definindo o Schema de Saída (Estrutura da Nota Fiscal) ---

# Sub-estrutura para cada Item da Nota
class ItemNota(BaseModel):
    descricao: str = Field(description="Nome ou descrição completa do produto/serviço.")
    quantidade: float = Field(description="Quantidade do item, convertida para um valor numérico (float).")
    valor_unitario: float = Field(description="Valor unitário do item.")
    valor_total: float = Field(description="Valor total da linha do item.")
    codigo_cfop: str = Field(description="Código CFOP (Natureza da Operação) associado ao item, se disponível.")
    cst_csosn: str = Field(description="Código CST (Situação Tributária) ou CSOSN do item, se disponível.")
    valor_aprox_tributos: float = Field(description="Valor aproximado dos tributos incidentes sobre este item (Lei da Transparência).")

# Sub-estrutura para Emitente e Destinatário
class ParteFiscal(BaseModel):
    cnpj_cpf: str = Field(description="CNPJ ou CPF da parte fiscal (apenas dígitos).")
    
    nome_razao: str = Field(
        description="Nome ou Razão Social completa."
    )
    
    endereco_completo: str = Field(description="Endereço completo (Rua, Número, Bairro, Cidade, Estado).")
    inscricao_estadual: str = Field(description="Inscrição Estadual, se disponível.")

# Sub-estrutura para os Totais de Impostos (Nível de Nota - Híbrido)
class TotaisImposto(BaseModel):
    base_calculo_icms: float = Field(description="Valor total da Base de Cálculo do ICMS da nota.")
    valor_total_icms: float = Field(description="Valor total do ICMS destacado na nota.")
    valor_total_ipi: float = Field(description="Valor total do IPI destacado na nota.")
    valor_total_pis: float = Field(description="Valor total do PIS destacado na nota.")
    valor_total_cofins: float = Field(description="Valor total do COFINS destacado na nota.")
    valor_outras_despesas: float = Field(description="Valor total de outras despesas acessórias (frete, seguro, etc.).")
    valor_aprox_tributos: float = Field(description="Valor aproximado total dos tributos.") 

# Estrutura Principal da Nota Fiscal
class NotaFiscal(BaseModel):
    """Estrutura Padrão e Completa dos Dados de uma Nota Fiscal."""
    
    chave_acesso: str = Field(description="Chave de Acesso da NF-e (44 dígitos), se presente.")
    modelo_documento: str = Field(description="Modelo do documento fiscal (Ex: NF-e, NFS-e, Cupom).")
    data_emissao: str = Field(description="Data de emissão da nota fiscal no formato YYYY-MM-DD.")
    valor_total_nota: float = Field(description="Valor total FINAL da nota fiscal (somatório de tudo).")
    natureza_operacao: str = Field(description="Descrição da natureza da operação (Ex: Venda de Mercadoria, Remessa para Armazém Geral).")
    
    emitente: ParteFiscal = Field(description="Dados completos do emitente (quem vendeu/prestou o serviço).")
    destinatario: ParteFiscal = Field(description="Dados completos do destinatário (quem comprou/recebeu o serviço).")
    
    totais_impostos: TotaisImposto = Field(description="Valores totais de impostos e despesas acessórias da nota.")

    itens: list[ItemNota] = Field(description="Lista completa de todos os produtos ou serviços discriminados na nota, seguindo o esquema ItemNota.")


# --- Função de Parsing de XML (Novo) ---
def parse_xml_nfe(xml_content: str) -> dict:
    """
    Processa o conteúdo XML de uma NF-e e extrai os dados diretamente 
    para o formato de dicionário compatível com NotaFiscal.
    """
    
    # 1. Parsing do XML
    # Remove o namespace para facilitar o XPath
    xml_content = xml_content.replace('xmlns="http://www.portalfiscal.inf.br/nfe"', '')
    root = ET.fromstring(xml_content)
    
    # Define a função de busca segura (XPath simples)
    def find_text(path, element=root, default=""):
        node = element.find(path)
        return node.text if node is not None else default

    def safe_float(text):
        try:
            # Substitui vírgula por ponto para parsing
            if isinstance(text, str):
                 text = text.replace(',', '.') 
            return float(text)
        except (ValueError, TypeError):
            return 0.0
    
    # 2. Dados de Cabeçalho (ide) e Totais
    
    # Caminho base para os dados da NF
    infNFe = root.find('.//infNFe')
    
    # Dados Principais
    chave_acesso = find_text('.//chNFe')
    if not chave_acesso:
         # Tenta a chave em Id
         chave_acesso = find_text('.//Id', default="").replace('NFe', '')
    
    data_emissao = find_text('.//dhEmi') # datetime ISO
    if not data_emissao:
        data_emissao = find_text('.//dEmi') # date YYYY-MM-DD
    
    # Ajusta a data para YYYY-MM-DD
    if data_emissao and len(data_emissao) > 10:
        data_emissao = data_emissao[:10]
        
    modelo_documento = find_text('.//mod')
    valor_total_nota = safe_float(find_text('.//vNF'))
    natureza_operacao = find_text('.//natOp')

    # Totais de Impostos (imposto/ICMSTot)
    icms_tot = root.find('.//ICMSTot')
    totais_impostos = {
        'base_calculo_icms': safe_float(find_text('.//vBC', icms_tot)),
        'valor_total_icms': safe_float(find_text('.//vICMS', icms_tot)),
        'valor_total_ipi': safe_float(find_text('.//vIPI', icms_tot)),
        'valor_total_pis': safe_float(find_text('.//vPIS', icms_tot)),
        'valor_total_cofins': safe_float(find_text('.//vCOFINS', icms_tot)),
        'valor_outras_despesas': safe_float(find_text('.//vOutro', icms_tot)),
        # Valor aproximado dos tributos (vTotTrib)
        'valor_aprox_tributos': safe_float(find_text('.//vTotTrib', icms_tot)),
    }

    # 3. Emitente (emit) e Destinatário (dest)

    def extract_parte_fiscal(element_tag):
        element = root.find(f'.//{element_tag}')
        if element is None: return {}

        cnpj_cpf = find_text('.//CNPJ', element) or find_text('.//CPF', element)
        
        ender = element.find('.//enderEmit') or element.find('.//enderDest')
        
        endereco_completo = ""
        if ender is not None:
             logradouro = find_text('.//xLgr', ender)
             numero = find_text('.//nro', ender)
             bairro = find_text('.//xBairro', ender)
             municipio = find_text('.//xMun', ender)
             uf = find_text('.//UF', ender)
             endereco_completo = f"{logradouro}, {numero} - {bairro} - {municipio}/{uf}".strip() if all([logradouro, numero, municipio, uf]) else ""

        return {
            'cnpj_cpf': cnpj_cpf,
            'nome_razao': find_text('.//xNome', element),
            'endereco_completo': endereco_completo,
            'inscricao_estadual': find_text('.//IE', element),
        }

    emitente = extract_parte_fiscal('emit')
    destinatario = extract_parte_fiscal('dest')

    # 4. Itens (det)
    itens = []
    for det in root.findall('.//det'):
        prod = det.find('.//prod')
        imposto = det.find('.//imposto')
        
        # Extração de CST/CSOSN
        cst_csosn = ""
        icms_node = imposto.find('.//ICMS')
        if icms_node is not None:
            for icms_subnode in icms_node:
                if 'CST' in icms_subnode.tag:
                    cst_csosn = find_text('.//CST', icms_subnode)
                    break
                elif 'CSOSN' in icms_subnode.tag:
                    cst_csosn = find_text('.//CSOSN', icms_subnode)
                    break
        
        # Extração do vTotTrib para o item
        v_aprox_tributos = 0.0
        if imposto.find('.//impostoTrib') is not None:
             v_aprox_tributos = safe_float(find_text('.//vTotTrib', imposto.find('.//impostoTrib')))

        itens.append({
            'descricao': find_text('.//xProd', prod),
            'quantidade': safe_float(find_text('.//qCom', prod)),
            'valor_unitario': safe_float(find_text('.//vUnCom', prod)),
            'valor_total': safe_float(find_text('.//vProd', prod)),
            'codigo_cfop': find_text('.//CFOP', prod),
            'cst_csosn': cst_csosn,
            'valor_aprox_tributos': v_aprox_tributos,
        })


    # 5. Montagem do Resultado Final
    result = {
        'chave_acesso': chave_acesso,
        'modelo_documento': modelo_documento,
        'data_emissao': data_emissao,
        'valor_total_nota': valor_total_nota,
        'natureza_operacao': natureza_operacao,
        'emitente': emitente,
        'destinatario': destinatario,
        'totais_impostos': totais_impostos,
        'itens': itens,
    }
    
    return result


# --- Função de Checagem de Qualidade ---
def check_for_missing_data(data_dict: dict) -> list:
    """Verifica se há dados críticos faltantes ou zerados e retorna uma lista de avisos."""
    warnings = []
    
    emitente = data_dict.get('emitente', {})
    destinatario = data_dict.get('destinatario', {})

    if not emitente.get('cnpj_cpf') or not emitente.get('nome_razao'):
        warnings.append("❌ Dados completos do Emitente estão faltando ou ilegíveis.")
    
    if not destinatario.get('cnpj_cpf') or not destinatario.get('nome_razao'):
        warnings.append("❌ Dados completos do Destinatário estão faltando ou ilegíveis.")

    valor_total_nota = data_dict.get('valor_total_nota', 0.0)
    if valor_total_nota <= 0.0:
        warnings.append("❌ O 'Valor Total da Nota' está zerado (R$ 0,00).")
    
    if not data_dict.get('itens'):
        warnings.append("❌ A lista de Itens/Produtos está vazia.")
    
    return warnings

# --- Função Central de OCR (Lida com Imagem e PDF) ---
def extract_text_from_file(uploaded_file):
    """
    Processa o arquivo carregado (JPG/PNG ou PDF) e retorna o texto extraído
    usando Tesseract OCR, com pré-processamento do OpenCV.
    Se for PDF multipágina, concatena o texto de todas as páginas.
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)
    
    tesseract_config = '--psm 4' 
    
    # Lista para armazenar o texto de cada página
    full_text_list = []
    
    # Lista de imagens a serem processadas
    images_to_process = []
    img_to_display = None
    
    # 1. Se for PDF (Múltiplas páginas)
    if "pdf" in file_type:
        try:
            # Converte TODAS as páginas (last_page=1 foi removido)
            images_to_process = convert_from_bytes(uploaded_file.read())
            
            if not images_to_process:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."
            
            # Armazena apenas a primeira imagem para visualização na sidebar
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
                # 1. Pré-processamento (OpenCV)
                # A checagem de qualidade/alerta é feita na primeira página
                img_for_ocr = preprocess_image_for_ocr(image_pil)
                
                # 2. Executa o OCR no array numpy processado
                text = pytesseract.image_to_string(img_for_ocr, lang='por', config=tesseract_config)
                
                # Adiciona o texto com um separador
                full_text_list.append(f"\n--- INÍCIO PÁGINA {i+1} ---\n\n" + text)
            
            # Salva a primeira imagem para exibição na sidebar (se não for nula)
            if img_to_display is not None:
                st.session_state["image_to_display"] = img_to_display 
            
            # Retorna o texto concatenado
            return "\n".join(full_text_list)

        except pytesseract.TesseractNotFoundError:
            return "ERRO_IMAGEM: O Tesseract não está instalado corretamente via packages.txt."
        except Exception as e:
            return f"ERRO_PROCESSAMENTO: Falha no OCR ou pré-processamento. Detalhes: {e}"
            
    return "ERRO_FALHA_GERAL: Falha desconhecida na extração de texto."

# --- Função de Enriquecimento e Pós-Validação ---
def enrich_and_validate_extraction(data_dict: dict, ocr_text: str) -> tuple[dict, list]:
    """
    1. Executa fallback heurístico (Regex) para CFOP/CST/CSOSN em itens.
    2. Executa pós-validação comparando o total de itens com o total da nota.
    Retorna o dicionário enriquecido e uma lista de mensagens para exibição.
    """
    
    enriched_data = data_dict.copy()
    itens_processados = []
    total_itens_calculado = 0.0
    messages = [] # Lista de mensagens de auditoria para retorno
    
    # Padrões de Regex (CFOP e CST/CSOSN)
    # CFOP: 4 dígitos obrigatórios (Ex: 5102)
    cfop_pattern = re.compile(r'\b(\d{4})\b')
    # CST/CSOSN: 2 a 3 dígitos (Ex: 00, 102)
    cst_pattern = re.compile(r'\b(0\d{2}|[1-9]\d{1,2})\b')
    
    # 1. Fallback Heurístico (Regex) para Itens
    if ocr_text:
        messages.append(("info", "Iniciando enriquecimento heurístico para códigos fiscais (CFOP, CST/CSOSN)."))
        
        for item in enriched_data.get('itens', []):
            item_desc_lower = item['descricao'].lower()
            
            # Converte valores para float e soma o total
            try:
                item['valor_total'] = float(item['valor_total'])
            except (TypeError, ValueError):
                 item['valor_total'] = 0.0
                 
            total_itens_calculado += item['valor_total']

            
            # Fallback para CFOP
            if not item.get('codigo_cfop') or len(item['codigo_cfop']) != 4:
                # Busca CFOP na linha da descrição do item no texto bruto
                match = cfop_pattern.search(item_desc_lower)
                if match:
                    item['codigo_cfop'] = match.group(1)
                    messages.append(("success", f"✅ CFOP do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['codigo_cfop']}**"))


            # Fallback para CST/CSOSN
            if not item.get('cst_csosn') or len(item['cst_csosn']) < 2:
                # Busca CST/CSOSN na linha da descrição do item no texto bruto
                match = cst_pattern.search(item_desc_lower)
                if match:
                    item['cst_csosn'] = match.group(1)
                    messages.append(("success", f"✅ CST/CSOSN do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['cst_csosn']}**"))

            itens_processados.append(item)
    
    # Atualiza a lista de itens enriquecida
    enriched_data['itens'] = itens_processados
    
    
    # 2. Pós-validação de Totais
    
    messages.append(("info", "Iniciando pós-validação de consistência de totais."))
    
    valor_total_nota = enriched_data.get('valor_total_nota', 0.0)
    
    # Tolerância de 0.01 centavo (floating point errors)
    tolerance = 0.01 
    
    soma_itens_formatada = formatar_moeda_imp(total_itens_calculado)
    total_nf_formatado = formatar_moeda_imp(valor_total_nota)

    if abs(total_itens_calculado - valor_total_nota) <= tolerance:
        messages.append(("success", f"👍 **Consistência Aprovada!** O somatório dos itens é consistente com o Valor Total da Nota. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado}"))
    else:
        messages.append(("error", f"🚨 **ALERTA DE INCONSISTÊNCIA!** O somatório dos itens extraídos é diferente do Valor Total da Nota extraído. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado} | Recomendação: Verifique a qualidade do OCR ou edite os valores manualmente."))
        
    return enriched_data, messages

# --- Função Auxiliar: Checagem de Brilho ---
def get_image_brightness(image_np):
    """Calcula o brilho médio da imagem (escala de cinza)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# --- Função Central: Pré-processamento de Imagem (OpenCV) ---
def preprocess_image_for_ocr(image_pil: Image.Image) -> np.ndarray:
    """
    Aplica pré-processamento (OpenCV) para aumentar a robustez do OCR.
    Passos: Binarização adaptativa e Remoção de Ruído.
    """
    
    # 1. Converte PIL Image para array numpy (BGR)
    image_np = np.array(image_pil.convert('RGB'))
    image_np = image_np[:, :, ::-1].copy() # Converte RGB para BGR (formato OpenCV)
    
    # 2. Converte para Escala de Cinza
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # 3. Alerta básico de qualidade (Brilho)
    brightness = get_image_brightness(image_np)
    if brightness < 80:
        st.sidebar.warning(f"⚠️ Nota: Imagem escura (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")
    elif brightness > 220:
         st.sidebar.warning(f"⚠️ Nota: Imagem muito clara (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")
         
    # 4. Suavização (Remoção de Ruído)
    # A mediana é boa para ruído de sal e pimenta (digitalizações ruins)
    denoised = cv2.medianBlur(gray, 3) 
    
    # 5. Binarização Adaptativa (Melhor para diferentes níveis de iluminação)
    # Garante que texto em áreas claras e escuras seja extraído
    processed_image = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, # Tamanho do bloco
        2   # Constante subtraída
    )
    
    return processed_image

# --- Funções Auxiliares ---
def formatar_moeda_imp(valor):
    """Função auxiliar para formatar float como moeda brasileira (R$ X.XXX,XX)."""
    if valor is None or valor == 0.0:
        return "R$ 0,00"
    # Lógica: substitui vírgula por X, ponto por vírgula, X por ponto.
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# --- Configuração do Modelo Gemini ---
llm = None
if "google_api_key" in st.secrets:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets["google_api_key"],
            temperature=0.1
        )
        if "llm_ready" not in st.session_state:
             st.session_state["llm_ready"] = True
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo Gemini. Detalhes: {e}")
        st.session_state["llm_ready"] = False
else:
    st.session_state["llm_ready"] = False


# --- FUNÇÃO DE EXIBIÇÃO DE RESULTADOS (UNIFICADA) ---
def display_extraction_results(data_dict: dict, source: str, ocr_text: Optional[str] = None):
    """Exibe os resultados estruturados na tela principal, independentemente da fonte (XML ou LLM), e o Dashboard."""
    
    st.header(f"✅ Resultado da Extração Estruturada ({source})")
    
    # 1. Pós-validação (Reaproveitando a chamada para exibir os alertas do Ponto 3)
    # A validação e o enriquecimento precisam ser feitos primeiro
    if source == "LLM/OCR" and ocr_text:
        # PONTO CHAVE: Recebe o dicionário enriquecido E a lista de mensagens de auditoria
        data_dict, audit_messages = enrich_and_validate_extraction(data_dict, ocr_text) 
        
        st.markdown("---")
        st.subheader("🛠️ Enriquecimento e Auditoria Pós-Extração")
        
        # Itera sobre as mensagens e as exibe APENAS UMA VEZ
        for msg_type, msg_text in audit_messages:
            # Renderiza as mensagens usando os comandos Streamlit corretos e ícones
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
    total_tributos = impostos_data.get('valor_aprox_tributos', 0.0) # Assume-se que o Ponto 3 populou o melhor valor
    total_icms = impostos_data.get('valor_total_icms', 0.0)
    total_ipi = impostos_data.get('valor_total_ipi', 0.0)
    
    # Colunas para KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    # KPI 1: Valor Total da Nota
    kpi1.metric("Valor Total da NF", formatar_moeda_imp(valor_total).replace("R$ ", ""))
    
    # KPI 2: Total de Tributos
    kpi2.metric("V. Aprox. Tributos", formatar_moeda_imp(total_tributos).replace("R$ ", ""))

    # KPI 3: Total de ICMS
    kpi3.metric("Total ICMS", formatar_moeda_imp(total_icms).replace("R$ ", ""))
    
    # KPI 4: Total de Itens
    kpi4.metric("Nº de Itens", total_itens)
    
    # KPI 5: Inconsistências
    kpi5.metric("Inconsistências", len(quality_warnings), delta="Críticas encontradas", delta_color="inverse")
    
    st.markdown("---")
    
    
    # --- 4. DETALHES GERAIS DA NOTA ---
    st.subheader("Informações Principais")
    
    # Cabeçalho da Nota (mantendo o st.columns original)
    col_data, col_valor, col_modelo, col_natureza = st.columns(4)
    
    col_data.metric("Data de Emissão", data_dict['data_emissao'])
    
    # Reutiliza o valor total formatado, mas sem a label redundante
    col_valor.metric("Valor Total da Nota", formatar_moeda_imp(data_dict.get('valor_total_nota', 0.0)).replace("R$ ", "")) 
    
    col_modelo.metric("Modelo Fiscal", data_dict['modelo_documento'])
    col_natureza.metric("Natureza da Operação", data_dict['natureza_operacao'])


    st.markdown("---")
    
    # Chave de Acesso
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


    # 6. Tabela de Itens
    st.subheader("🛒 Itens da Nota Fiscal")
    
    itens_list = data_dict.get('itens', [])
    
    if itens_list:
        
        # Cria DataFrame para Tabela e Gráfico
        df_itens = pd.DataFrame(itens_list)
        
        # Garante que as colunas numéricas estejam no formato correto para o gráfico
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
        
        # --- NOVO: Gráfico de Distribuição por CFOP (Ponto 5) ---
        st.markdown("### 📈 Distribuição de Valor por CFOP")
        # Agrupa pelo CFOP e soma o valor total
        df_cfop = df_itens.groupby('codigo_cfop', dropna=False)['valor_total'].sum().reset_index()
        df_cfop.columns = ['CFOP', 'Valor Total']
        
        # Cria o gráfico de barras interativo (Plotly)
        fig = px.bar(
            df_cfop, 
            x='CFOP', 
            y='Valor Total', 
            text='Valor Total',
            labels={'Valor Total': 'Valor Total (R$)', 'CFOP': 'Código Fiscal de Operações'},
            color='CFOP',
            title='Valor de Produtos/Serviços agrupado por CFOP'
        )
        # Formata o texto do gráfico para moeda
        fig.update_traces(texttemplate='R$ %{y:,.2f}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        
    else:
        st.warning("Nenhum item ou serviço foi encontrado na nota fiscal.")


    # 7. Exibição dos Totais de Impostos (Mantido)
    st.markdown("---")
    st.subheader("💰 Totais de Impostos e Despesas")

    # A lógica de cálculo/desempate de tributos é mantida, mas agora usa o dicionário final
    
    total_tributos_calculado = df_itens['valor_aprox_tributos'].sum() if 'df_itens' in locals() else 0.0
    total_tributos_extraido_direto = impostos_data.get('valor_aprox_tributos', 0.0)

    # LÓGICA DE DESEMPATE CRÍTICA:
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
    
    
    # 8. Edição Manual Assistida (Mantido)
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
            # Atualiza o data_dict para o download
            data_dict['totais_impostos']['valor_total_icms'] = float(icms_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_ipi'] = float(ipi_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_pis'] = float(pis_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_cofins'] = float(cofins_manual.replace(",", "."))
            st.success("Valores de impostos atualizados para o JSON de download.")
            
        except ValueError:
            st.error("Por favor, insira apenas números válidos nos campos de edição.")
            
        # Re-atualiza o dicionário final com os inputs manuais
        final_data_dict = data_dict


    # --- 9. Botões de Download (JSON e NOVO: CSV) ---
    st.markdown("---")
    col_json_btn, col_csv_btn = st.columns(2)
    
    try:
        nome_curto = data_dict['emitente']['nome_razao'].split(' ')[0]
        data_emissao_nome = data_dict['data_emissao']
    except (KeyError, IndexError, TypeError):
        nome_curto = "extraida"
        data_emissao_nome = "data_desconhecida"

    # Botão de Download JSON
    json_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
    col_json_btn.download_button(
        label="⬇️ Baixar JSON COMPLETO da Extração",
        data=json_data,
        file_name=f"nf_{data_emissao_nome}_{nome_curto}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # Botão de Download CSV (Apenas dos Itens)
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


# --- Configuração da Interface Streamlit ---
st.set_page_config(page_title="Extrator Autonometa", layout="wide")
st.title("🤖 Extrator Autonometa (OCR/XML + LLM) de Notas Fiscais")
st.markdown("---")

# --- Logo na Sidebar ---
st.sidebar.image("https://i.imgur.com/oH1wbZ4.png")

# --- 1. Botão de Carregamento na Sidebar ---
st.sidebar.header("Upload da Nota Fiscal (1/2)")

uploaded_file = st.sidebar.file_uploader(
    "Escolha a Nota Fiscal (JPG, PNG, PDF ou XML):",
    type=['png', 'jpg', 'jpeg', 'pdf', 'xml']
)

parser = PydanticOutputParser(pydantic_object=NotaFiscal)


if uploaded_file is not None:
    
    file_type = uploaded_file.type
    # Reseta os estados de processamento
    st.session_state["extracted_data_xml"] = None 
    st.session_state["xml_processed"] = False
    
    # --- NOVO: PROCESSAMENTO DE XML (PRIORITÁRIO) ---
    if "xml" in file_type or uploaded_file.name.lower().endswith('.xml'):
        st.sidebar.success("Arquivo XML detectado.")
        
        try:
            # Lê o conteúdo como string
            uploaded_file.seek(0)
            xml_content = uploaded_file.getvalue().decode('utf-8')
            
            with st.spinner("Analisando e estruturando dados do XML..."):
                # Chama a nova função de parsing
                xml_data_dict = parse_xml_nfe(xml_content)
                
                # Validação Pydantic para garantir que o XML segue o schema
                NotaFiscal(**xml_data_dict)
                
                st.session_state["extracted_data_xml"] = xml_data_dict
                st.session_state["xml_processed"] = True # Sinaliza sucesso
            
            st.sidebar.info("Dados extraídos diretamente do XML com sucesso!")
            
            # Chama a exibição imediata
            display_extraction_results(xml_data_dict, source="XML")

        except Exception as e:
            st.error(f"Erro ao processar o arquivo XML. O arquivo pode estar malformado ou não seguir o schema NF-e. Detalhes: {e}")
            st.session_state["xml_processed"] = False
            
    # --- PROCESSAMENTO DE IMAGEM/PDF (OCR + LLM) ---
    else:
        # 1. Executa a extração do texto bruto (OCR)
        with st.spinner("Extraindo texto bruto da nota fiscal (OCR)..."):
            ocr_text = extract_text_from_file(uploaded_file)
            
        st.session_state["ocr_text"] = ocr_text
    
        # --- 2. Miniatura da Imagem na Sidebar ---
        if "image_to_display" in st.session_state:
            st.sidebar.success("Imagem carregada e OCR inicial concluído.")
            with st.sidebar.expander("🔎 Visualizar Nota Fiscal"):
                st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", width='stretch')
        else:
            # Exibir erro de OCR na sidebar se houver
            if "ERRO" in st.session_state.get("ocr_text", ""):
                st.sidebar.error(f"Erro no OCR: {st.session_state['ocr_text']}")
            else:
                st.sidebar.info("Arquivo PDF processado. Clique para continuar.")
        

        # 3. Próxima Etapa: Botão de Interpretação LLM 
        if "ERRO" not in st.session_state.get("ocr_text", ""):
            st.subheader("Interpretação de Dados Estruturados (2/2)")
            
            if st.session_state.get("llm_ready", False):
                if st.button("🚀 Interpretar Dados Estruturados com o Agente Gemini", key="run_extraction_btn"):
                    st.session_state["run_llm_extraction"] = True
                    st.rerun()
            else:
                st.error("⚠️ O Agente Gemini não está pronto. Verifique sua `google_api_key`.")


# --- Seção de Execução da Extração (LLM - Execução Inline) ---
if st.session_state.get("run_llm_extraction", False) and st.session_state.get("llm_ready", False):
    
    st.session_state["run_llm_extraction"] = False 
    
    text_to_analyze = st.session_state.get("ocr_text", "")
    response = None 
    
    if not text_to_analyze or "ERRO" in text_to_analyze:
        st.error("Não há texto válido para enviar ao Agente LLM.")
        st.stop()

    # Início do bloco de execução original do LLM
    try:
        with st.spinner("⏳ O Agente Gemini está interpretando o texto (o tempo de resposta é de aproximadamente 1 minuto)..."):
            
            # 2. Criando o Prompt de Extração de Texto (Prompt Atualizado)
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", 
                        "Você é um agente de extração de dados fiscais. Sua tarefa é analisar o texto bruto de uma nota fiscal e extrair TODAS as informações solicitadas no formato JSON. "
                        "Instruções Específicas: Garanta a extração do campo `natureza_operacao` e da `chave_acesso`. "
                        "ATENÇÃO HÍBRIDA: Para o Valor Aproximado dos Tributos, primeiro tente preencher o campo `valor_aprox_tributos` DENTRO DE CADA ITEM. Se essa informação estiver ausente na tabela de itens, procure o valor TOTAL no campo de 'Dados Adicionais' e preencha o campo `totais_impostos.valor_aprox_tributos`."
                        "Converta todos os valores monetários e numéricos para float. Não invente dados."
                    ),
                    
                    ("human", (
                        "Analise o texto a seguir e extraia os campos fiscais na estrutura JSON. "
                        "Instrução Fiscal Crítica: Priorize a extração do valor de tributos item por item. Se não houver, extraia o total dos tributos do campo de Dados Adicionais."
                        "Obrigatório: extraia a lista de itens APENAS DA TABELA PRINCIPAL.\n\n"
                        "INSTRUÇÕES DE FORMATO:\n"
                        "{format_instructions}\n\n"
                        "TEXTO BRUTO DA NOTA:\n"
                        "{text_to_analyze}"
                    )),
                ]
            )
            
            prompt_values = prompt_template.partial(
                format_instructions=parser.get_format_instructions()
            )
            
            final_prompt = prompt_values.format_messages(text_to_analyze=text_to_analyze)

            # 3. Execução do LLM
            response = llm.invoke(final_prompt)
            extracted_data = parser.parse(response.content)
            
        # --- NOVO: Enriquecimento e Pós-Validação ---
        data_dict = extracted_data.model_dump()
        
        # Chama a nova função com o dicionário e o texto bruto
        final_data_dict = enrich_and_validate_extraction(data_dict, text_to_analyze)
            
        # CHAMA A FUNÇÃO DE DISPLAY APÓS SUCESSO DO LLM E ENRIQUECIMENTO
        display_extraction_results(data_dict, source="LLM/OCR", ocr_text=text_to_analyze)

    except ValidationError as ve:
        st.error("Houve um erro de validação (Pydantic). O Gemini pode ter retornado um JSON malformado.")
        if response is not None:
            with st.expander("Ver Resposta Bruta do LLM (JSON malformado)", expanded=True):
                st.code(response.content, language='json')
        st.warning(f"Detalhes do Erro: {ve}")

    except Exception as e:
        st.error(f"Houve um erro geral durante a interpretação pelo Gemini. Detalhes: {e}")
        if 'response' in locals() and response is not None:
            with st.expander("Ver Resposta Bruta do LLM (Debugging)", expanded=True):
                st.code(response.content, language='json')
        
st.markdown("---")
