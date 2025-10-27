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


# =======================================================================
# --- 1. CONFIGURAÇÕES INICIAIS E GERAIS ---
# =======================================================================

# Configuração do Streamlit
st.set_page_config(
    page_title="Extrator Autonometa",
    layout="wide",
    initial_sidebar_state="auto"
)

# Configuração do Tesseract
TESSERACT_PATH = '/usr/bin/tesseract'
if 'TESSERACT_PATH' in os.environ:
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_PATH']
elif os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    pass

# Inicialização do LLM (lida com chaves via st.secrets)
llm = None
if "google_api_key" in st.secrets:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets["google_api_key"],
            temperature=0.1
        )
        st.session_state["llm_ready"] = True
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo Gemini. Detalhes: {e}")
        st.session_state["llm_ready"] = False
else:
    st.session_state["llm_ready"] = False

if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None

if "file_uploader_key_id" not in st.session_state:
    st.session_state["file_uploader_key_id"] = 0

# =======================================================================
# --- 2. DEFININDO OS SCHEMAS DE SAÍDA (ESTRUTURAS PYDANTIC) ---
# =======================================================================

class ItemNota(BaseModel):
    """Sub-estrutura para cada Item da Nota Fiscal."""
    descricao: str = Field(description="Nome ou descrição completa do produto/serviço.")
    quantidade: float = Field(description="Quantidade do item, convertida para um valor numérico (float).")
    valor_unitario: float = Field(description="Valor unitário do item.")
    valor_total: float = Field(description="Valor total da linha do item.")
    codigo_cfop: str = Field(description="Código CFOP (Natureza da Operação) associado ao item, se disponível.")
    cst_csosn: str = Field(description="Código CST (Situação Tributária) ou CSOSN do item, se disponível.")
    valor_aprox_tributos: float = Field(description="Valor aproximado dos tributos incidentes sobre este item (Lei da Transparência).")


class ParteFiscal(BaseModel):
    """Sub-estrutura para Emitente e Destinatário."""
    cnpj_cpf: str = Field(description="CNPJ ou CPF da parte fiscal (apenas dígitos).")
    nome_razao: str = Field(
        description="Nome ou Razão Social completa.",
    )
    endereco_completo: str = Field(description="Endereço completo (Rua, Número, Bairro, Cidade, Estado).")
    inscricao_estadual: str = Field(description="Inscrição Estadual, se disponível.")


class TotaisImposto(BaseModel):
    """Sub-estrutura para os Totais de Impostos (Nível de Nota)."""
    base_calculo_icms: float = Field(description="Valor total da Base de Cálculo do ICMS da nota.")
    valor_total_icms: float = Field(description="Valor total do ICMS destacado na nota.")
    valor_total_ipi: float = Field(description="Valor total do IPI destacado na nota.")
    valor_total_pis: float = Field(description="Valor total do PIS destacado na nota.")
    valor_total_cofins: float = Field(description="Valor total do COFINS destacado na nota.")
    valor_outras_despesas: float = Field(description="Valor total de outras despesas acessórias (frete, seguro, etc.).")
    valor_aprox_tributos: float = Field(description="Valor aproximado total dos tributos.")


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


# =======================================================================
# --- 3. FUNÇÕES DE EXTRAÇÃO, PRÉ-PROCESSAMENTO E AUXILIARES ---
# =======================================================================

def formatar_moeda_imp(valor):
    """Função auxiliar para formatar float como moeda brasileira (R$ X.XXX,XX)."""
    if valor is None or valor == 0.0:
        return "R$ 0,00"
    try:
        # Lógica: substitui vírgula por X, ponto por vírgula, X por ponto.
        return f"R$ {float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "R$ 0,00"


def get_image_brightness(image_np):
    """Calcula o brilho médio da imagem (escala de cinza)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def preprocess_image_for_ocr(image_pil: Image.Image) -> np.ndarray:
    """
    Aplica pré-processamento (OpenCV) para aumentar a robustez do OCR.
    Passos: Binarização adaptativa e Remoção de Ruído.
    """
    image_np = np.array(image_pil.convert('RGB'))
    image_np = image_np[:, :, ::-1].copy()

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Alerta básico de qualidade (Brilho)
    brightness = get_image_brightness(image_np)
    if brightness < 80:
        st.sidebar.warning(f"⚠️ Nota: Imagem escura (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")
    elif brightness > 220:
         st.sidebar.warning(f"⚠️ Nota: Imagem muito clara (Brilho: {brightness:.0f}). A precisão do OCR pode ser afetada.")

    # Suavização (Remoção de Ruído)
    denoised = cv2.medianBlur(gray, 3)

    # Binarização Adaptativa
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
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)

    tesseract_config = '--psm 4'

    full_text_list = []
    images_to_process = []
    img_to_display = None

    if "pdf" in file_type:
        try:
            images_to_process = convert_from_bytes(uploaded_file.read())

            if not images_to_process:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."

            img_to_display = images_to_process[0]

        except Exception as e:
            return f"ERRO_PDF: Verifique se 'poppler-utils' está instalado via packages.txt. Detalhes: {e}"

    elif "image" in file_type:
        try:
            img_to_display = Image.open(uploaded_file)
            images_to_process.append(img_to_display)
        except Exception as e:
            return f"ERRO_IMAGEM: Falha na abertura da imagem. Detalhes: {e}"

    else:
        return "ERRO_TIPO_INVALIDO: Tipo de arquivo não suportado."

    if images_to_process:
        try:
            for i, image_pil in enumerate(images_to_process):
                img_for_ocr = preprocess_image_for_ocr(image_pil)
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


def parse_xml_nfe(xml_content: str) -> dict:
    """
    Processa o conteúdo XML de uma NF-e e extrai os dados diretamente
    para o formato de dicionário compatível com NotaFiscal.
    """
    xml_content = xml_content.replace('xmlns="http://www.portalfiscal.inf.br/nfe"', '')
    root = ET.fromstring(xml_content)

    def find_text(path, element=root, default=""):
        node = element.find(path)
        return node.text if node is not None else default

    def safe_float(text):
        try:
            if isinstance(text, str):
                 text = text.replace(',', '.')
            return float(text)
        except (ValueError, TypeError):
            return 0.0

    # Dados Principais
    chave_acesso = find_text('.//chNFe') or find_text('.//Id', default="").replace('NFe', '')
    data_emissao = find_text('.//dhEmi') or find_text('.//dEmi')
    if data_emissao and len(data_emissao) > 10:
        data_emissao = data_emissao[:10]

    modelo_documento = find_text('.//mod')
    icms_tot = root.find('.//ICMSTot')
    valor_total_nota = safe_float(find_text('.//vNF', icms_tot))
    natureza_operacao = find_text('.//natOp')

    # Totais de Impostos (imposto/ICMSTot)
    totais_impostos = {
        'base_calculo_icms': safe_float(find_text('.//vBC', icms_tot)),
        'valor_total_icms': safe_float(find_text('.//vICMS', icms_tot)),
        'valor_total_ipi': safe_float(find_text('.//vIPI', icms_tot)),
        'valor_total_pis': safe_float(find_text('.//vPIS', icms_tot)),
        'valor_total_cofins': safe_float(find_text('.//vCOFINS', icms_tot)),
        'valor_outras_despesas': safe_float(find_text('.//vOutro', icms_tot)),
        'valor_aprox_tributos': safe_float(find_text('.//vTotTrib', icms_tot)),
    }

    # Emitente (emit) e Destinatário (dest)
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

    # Itens (det)
    itens = []
    for det in root.findall('.//det'):
        prod = det.find('.//prod')
        imposto = det.find('.//imposto')

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

    # Montagem do Resultado Final
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


def enrich_and_validate_extraction(data_dict: dict, ocr_text: str) -> tuple[dict, list]:
    """
    1. Executa fallback heurístico (Regex) para CFOP/CST/CSOSN em itens.
    2. Executa pós-validação comparando o total de itens com o total da nota.
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
            if not item.get('codigo_cfop') or len(item['codigo_cfop']) != 4:
                match = cfop_pattern.search(item_desc_lower)
                if match:
                    item['codigo_cfop'] = match.group(1)
                    messages.append(("success", f"✅ CFOP do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['codigo_cfop']}**"))

            # Fallback para CST/CSOSN
            if not item.get('cst_csosn') or len(item['cst_csosn']) < 2:
                match = cst_pattern.search(item_desc_lower)
                if match:
                    item['cst_csosn'] = match.group(1)
                    messages.append(("success", f"✅ CST/CSOSN do item '{item['descricao'][:20]}...' preenchido via Regex: **{item['cst_csosn']}**"))

            itens_processados.append(item)

        enriched_data['itens'] = itens_processados

    # 2. Pós-validação de Totais
    messages.append(("info", "Iniciando pós-validação de consistência de totais."))

    valor_total_nota = enriched_data.get('valor_total_nota', 0.0)
    tolerance = 0.01

    soma_itens_formatada = formatar_moeda_imp(total_itens_calculado)
    total_nf_formatado = formatar_moeda_imp(valor_total_nota)

    if abs(total_itens_calculado - valor_total_nota) <= tolerance:
        messages.append(("success", f"👍 **Consistência Aprovada!** O somatório dos itens é consistente com o Valor Total da Nota. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado}"))
    else:
        messages.append(("error", f"🚨 **ALERTA DE INCONSISTÊNCIA!** O somatório dos itens extraídos é diferente do Valor Total da Nota extraído. | Soma dos Itens: {soma_itens_formatada} | Total NF: {total_nf_formatado} | Recomendação: Verifique a qualidade do OCR ou edite os valores manualmente."))

    return enriched_data, messages


# =======================================================================
# --- 4. CONFIGURAÇÃO LLM E PROMPT ---
# =======================================================================

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Extraia os dados da nota fiscal no seguinte texto OCR. Retorne apenas o JSON. {format_instructions}\n\nTexto OCR:\n{text_to_analyze}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# =======================================================================
# --- 5. FUNÇÃO DE EXIBIÇÃO (STREAMLIT) ---
# =======================================================================

def display_extraction_results(data_dict: dict, source: str, ocr_text: Optional[str] = None):
    """Exibe os resultados estruturados na tela principal, independentemente da fonte (XML ou LLM), e o Dashboard."""

    st.header(f"✅ Resultado da Extração Estruturada ({source})")

    # 1. Pós-validação (Validação e Enriquecimento - Apenas para LLM/OCR)
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

    # 2. Checagem de Qualidade
    quality_warnings = check_for_missing_data(data_dict)

    if quality_warnings:
        # Usamos uma caixa de erro/aviso que é naturalmente mais visível
        st.error(f"⚠️ Atenção: Foram encontradas **{len(quality_warnings)} informações críticas faltando** ou zeradas na nota fiscal. Verifique a lista abaixo:")
        
        # O expander é forçado a abrir (expanded=True) se houver avisos
        with st.expander("Clique para ver os detalhes das inconsistências", expanded=True): 
            for warning in quality_warnings:
                st.markdown(warning)
    else:
        st.success("🎉 Verificação de Qualidade concluída: Nenhuma informação crítica obrigatória faltando (Emitente, Destinatário, Valor, Itens).")

    st.markdown("---") # Separador após a checagem

    # --- 2. DASHBOARD: KPIs e Indicadores ---
    st.subheader("📊 Resumo Fiscal (KPIs)")

    impostos_data = data_dict.get('totais_impostos', {})
    valor_total = data_dict.get('valor_total_nota', 0.0)
    total_itens = len(data_dict.get('itens', []))
    total_tributos = impostos_data.get('valor_aprox_tributos', 0.0)
    total_icms = impostos_data.get('valor_total_icms', 0.0)
    total_ipi = impostos_data.get('valor_total_ipi', 0.0)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4) # De 5 colunas para 4

    kpi1.metric("Valor Total da NF", formatar_moeda_imp(valor_total).replace("R$ ", ""))
    kpi2.metric("V. Aprox. Tributos", formatar_moeda_imp(total_tributos).replace("R$ ", ""))
    kpi3.metric("Total ICMS/IPI", f"{formatar_moeda_imp(total_icms).replace('R$ ', '')} / {formatar_moeda_imp(total_ipi).replace('R$ ', '')}")
    kpi4.metric("Nº de Itens", total_itens) 

    st.markdown("---")


    # --- 4. DETALHES GERAIS DA NOTA ---
    st.subheader("Informações Principais")

    col_data, col_valor, col_modelo, col_natureza = st.columns(4)

    col_data.metric("Data de Emissão", data_dict['data_emissao'])
    col_valor.metric("Valor Total da Nota", formatar_moeda_imp(data_dict.get('valor_total_nota', 0.0)).replace("R$ ", ""))
    col_modelo.metric("Modelo Fiscal", data_dict['modelo_documento'])
    with col_natureza:
        st.markdown("**Natureza da Operação**")
        st.info(data_dict['natureza_operacao'])


    st.markdown("---")

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
    df_itens = pd.DataFrame()

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

    total_tributos_calculado = df_itens['valor_aprox_tributos'].sum() if not df_itens.empty else 0.0
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
            # Atualiza o data_dict para o download
            data_dict['totais_impostos']['valor_total_icms'] = float(icms_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_ipi'] = float(ipi_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_pis'] = float(pis_manual.replace(",", "."))
            data_dict['totais_impostos']['valor_total_cofins'] = float(cofins_manual.replace(",", "."))
            st.success("Valores de impostos atualizados para o JSON de download.")
        except ValueError:
            st.error("Erro: Certifique-se de que os valores inseridos manualmente são números válidos.")

    # 9. Botões de Download (JSON e CSV)
    st.markdown("---")
    
    st.subheader("⬇️ Downloads")
    
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

    if not df_itens.empty:
        # 1. Renomeação das Colunas para Nomes Amigáveis (Português)
        df_csv = df_itens.rename(columns={
            "descricao": "Descricao_Produto",
            "quantidade": "Quantidade",
            "valor_unitario": "Valor_Unitario",
            "valor_total": "Valor_Total_Item",
            "codigo_cfop": "CFOP",
            "cst_csosn": "CST_CSOSN",
            "valor_aprox_tributos": "Valor_Aprox_Tributos"
        })

    # 2. Formatação dos Valores Financeiros/Decimais para Padrão Brasileiro (ponto -> vírgula)
    cols_para_formatar = ["Quantidade", "Valor_Unitario", "Valor_Total_Item", "Valor_Aprox_Tributos"]

    for col in cols_para_formatar:
        # Converte float para string no formato brasileiro com duas casas decimais
        df_csv[col] = df_csv[col].apply(lambda x: f"{x:.2f}".replace('.', ','))

    # 3. Geração do CSV no formato brasileiro (separador de ponto e vírgula)
    # Usamos 'sep=;' para evitar conflito com a vírgula decimal e adicionamos BOM para compatibilidade com Excel.
    csv_data = df_csv.to_csv(
        index=False,
        sep=';',
        encoding='utf-8-sig' # Adiciona Byte Order Mark para compatibilidade com Excel em PT
    )

    col_csv_btn.download_button(
        label="⬇️ Baixar Itens em CSV (Formato ABNT)",
        data=csv_data,
        file_name=f"itens_{data_emissao_nome}_{nome_curto}.csv",
        mime="text/csv",
        use_container_width=True
    )

    with st.expander("Ver JSON Bruto Completo (DEBUG)", expanded=False):
         st.json(data_dict)


# =======================================================================
# --- 6. LÓGICA PRINCIPAL DO APP (STREAMLIT) ---
# =======================================================================

st.title("Análise e Extração Estruturada de Dados 🧠")

if not st.session_state.get("llm_ready"):
    st.error("⚠️ Erro: A chave 'google_api_key' não foi encontrada nos secrets do Streamlit. O Extrator de PDF/Imagem (LLM/OCR) está desativado. Apenas a extração de XML está funcional.")

# --- Logo na Sidebar ---
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.imgur.com/oH1wbZ4.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# --- 1. Botão de Carregamento na Sidebar ---
st.sidebar.header("Upload da Nota Fiscal")

uploaded_file = st.sidebar.file_uploader(
    "Escolha a Nota Fiscal (JPG, PNG, PDF ou XML):",
    type=['png', 'jpg', 'jpeg', 'pdf', 'xml']
    key=st.session_state["file_uploader_key_id"]
)

if st.sidebar.button("🔄 Iniciar Novo Processo / Limpar", type='primary', use_container_width=True):
    # Lógica de Limpeza Completa
    if "processed_data" in st.session_state:
        del st.session_state["processed_data"]
    if "processed_source" in st.session_state:
        del st.session_state["processed_source"]
    if "ocr_text" in st.session_state:
        del st.session_state["ocr_text"]
    if "image_to_display" in st.session_state:
        del st.session_state["image_to_display"]

    st.session_state["file_uploader_key_id"] += 1
    
    st.rerun() # Força o reinício

if uploaded_file is not None:

    # 1. FLUXO DE EXIBIÇÃO: Se os dados já foram processados e estão no estado, exiba-os imediatamente.
    if st.session_state["processed_data"] is not None:
        data_dict = st.session_state["processed_data"]
        source = st.session_state["processed_source"]
        ocr_text = st.session_state.get("ocr_text", None) # Pega o OCR text se existir

        # Exibe os resultados sem re-processar
        display_extraction_results(data_dict, source=source, ocr_text=ocr_text)

    # 2. FLUXO DE PROCESSAMENTO: Se o arquivo foi carregado mas os dados AINDA NÃO estão no estado.
    elif st.session_state["processed_data"] is None:
        
        file_type = uploaded_file.type

        with st.spinner(f"Processando arquivo ({uploaded_file.name})..."):

            # --- FLUXO 1: XML (Prioridade Máxima) ---
            if "xml" in file_type:
                uploaded_file.seek(0)
                xml_content = uploaded_file.read().decode('utf-8')
                data_dict = parse_xml_nfe(xml_content)

                if "error" in data_dict:
                    st.error(data_dict["error"])
                else:
                    try:
                        NotaFiscal(**data_dict)
                        
                        # NOVO: SALVA NO ESTADO APÓS SUCESSO
                        st.session_state["processed_data"] = data_dict
                        st.session_state["processed_source"] = "XML"
                        
                        display_extraction_results(data_dict, source="XML")
                    except ValidationError as ve:
                        st.error(f"Erro de Validação Pydantic ao ler XML: {ve}")
                        st.info("O XML foi processado, mas falhou na validação do esquema Pydantic. Use o JSON Bruto para debug.")
                        display_extraction_results(data_dict, source="XML")

            # --- FLUXO 2: OCR/LLM (PDF/Imagem) ---
            elif st.session_state.get("llm_ready"):

                # 1. Extração de texto bruto (OCR)
                text_to_analyze = extract_text_from_file(uploaded_file)
                response = None

                if text_to_analyze.startswith("ERRO_"):
                     st.error(f"Erro na extração de texto (OCR): {text_to_analyze}")
                     st.markdown("Verifique se as dependências (poppler-utils, tesseract) estão instaladas corretamente.")
                else:
                    # 2. Miniatura da Imagem na Sidebar
                    if "image_to_display" in st.session_state:
                        st.sidebar.success("Imagem carregada e OCR inicial concluído.")
                        with st.sidebar.expander("🔎 Visualizar Nota Fiscal"):
                            st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", width='stretch')

                    try:
                        # 3. Execução do LLM
                        final_prompt = prompt.format(text_to_analyze=text_to_analyze)
                        response = llm.invoke(final_prompt)
                        extracted_data = parser.parse(response.content)

                        # 4. Pós-processamento e Enriquecimento
                        data_dict = extracted_data.model_dump()

                        # NOVO: SALVA NO ESTADO APÓS SUCESSO
                        st.session_state["processed_data"] = data_dict
                        st.session_state["processed_source"] = "LLM/OCR"
                        st.session_state["ocr_text"] = text_to_analyze # Salva o OCR text

                        # 5. CHAMA A FUNÇÃO DE DISPLAY
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
                             with st.expander("Ver Resposta Bruta do LLM", expanded=False):
                                st.code(response.content, language='text')
                        with st.expander("Ver Texto OCR Bruto"):
                            st.code(text_to_analyze, language="text")
            else:
                st.warning("O arquivo é uma imagem/PDF, mas o processamento LLM está desativado (sem Google API Key).")
            
# --- Fim do Código ---
