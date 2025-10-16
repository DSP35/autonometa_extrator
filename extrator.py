# Extrator Autonometa
# Desenvolvido por David Parede

import streamlit as st
from PIL import Image
import pytesseract         # OCR
from pdf2image import convert_from_bytes
from io import BytesIO
import json                # NECESSÁRIO para st.download_button
import os

# --- Imports LangChain e Pydantic ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
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
    
    # CORREÇÃO CRÍTICA: Adiciona alias 'nome_raza' para aceitar o erro comum do LLM
    nome_razao: str = Field(
        description="Nome ou Razão Social completa.",
        validation_alias='nome_raza'
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
    # Campo de retorno HÍBRIDO: O LLM preenche este campo se encontrar um total único no texto.
    valor_aprox_tributos: float = Field(description="Valor aproximado total dos tributos.") 

# Estrutura Principal da Nota Fiscal
class NotaFiscal(BaseModel):
    """Estrutura Padrão e Completa dos Dados de uma Nota Fiscal."""
    
    # Dados Gerais
    chave_acesso: str = Field(description="Chave de Acesso da NF-e (44 dígitos), se presente.")
    modelo_documento: str = Field(description="Modelo do documento fiscal (Ex: NF-e, NFS-e, Cupom).")
    data_emissao: str = Field(description="Data de emissão da nota fiscal no formato YYYY-MM-DD.")
    valor_total_nota: float = Field(description="Valor total FINAL da nota fiscal (somatório de tudo).")
    
    # Emitente e Destinatário
    emitente: ParteFiscal = Field(description="Dados completos do emitente (quem vendeu/prestou o serviço).")
    destinatario: ParteFiscal = Field(description="Dados completos do destinatário (quem comprou/recebeu o serviço).")
    
    # NOVOS TOTAIS DE IMPOSTOS
    totais_impostos: TotaisImposto = Field(description="Valores totais de impostos e despesas acessórias da nota.")

    # Itens/Serviços (Lista)
    itens: list[ItemNota] = Field(description="Lista completa de todos os produtos ou serviços discriminados na nota, seguindo o esquema ItemNota.")

# --- Função de Checagem de Qualidade ---
def check_for_missing_data(data_dict: dict) -> list:
    """Verifica se há dados críticos faltantes ou zerados e retorna uma lista de avisos."""
    warnings = []
    
    # 1. Checagem de CNPJ/Razão Social (o mais importante)
    emitente = data_dict.get('emitente', {})
    destinatario = data_dict.get('destinatario', {})

    if not emitente.get('cnpj_cpf') or not emitente.get('nome_razao'):
        warnings.append("❌ Dados completos do Emitente estão faltando ou ilegíveis.")
    
    if not destinatario.get('cnpj_cpf') or not destinatario.get('nome_razao'):
        warnings.append("❌ Dados completos do Destinatário estão faltando ou ilegíveis.")

    # 2. Checagem de Valores (Floats)
    valor_total_nota = data_dict.get('valor_total_nota', 0.0)
    if valor_total_nota <= 0.0:
        warnings.append("❌ O 'Valor Total da Nota' está zerado (R$ 0,00).")
    
    # 3. Checagem da lista de itens
    if not data_dict.get('itens'):
        warnings.append("❌ A lista de Itens/Produtos está vazia.")
    
    return warnings

# --- FUNÇÃO LLM CACHEADA (Para Otimização de Velocidade) ---
# CORREÇÃO: Usamos '_parser' para evitar o erro de hashing do Streamlit.
@st.cache_data(show_spinner="⏳ O Agente Gemini está interpretando o texto (a primeira extração pode levar até 1 minuto)...")
def run_llm_extraction_cached(llm_model: ChatGoogleGenerativeAI, text_to_analyze: str, _parser: PydanticOutputParser):
    """
    Executa a extração do LLM de forma cacheada.
    Se o 'text_to_analyze' for o mesmo, o resultado anterior é retornado instantaneamente.
    """
    # 2. Criando o Prompt de Extração de Texto com PromptTemplate robusto
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", 
                "Você é um agente de extração de dados fiscais. Sua tarefa é analisar o texto bruto de uma nota fiscal e extrair TODAS as informações solicitadas no formato JSON. "
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
        format_instructions=_parser.get_format_instructions() # Usamos _parser aqui
    )
    
    final_prompt = prompt_values.format_messages(text_to_analyze=text_to_analyze)

    # 3. Execução do LLM
    response = llm_model.invoke(final_prompt)
    extracted_data = _parser.parse(response.content) # E aqui

    # Retorna a resposta bruta (para debug) e os dados validados
    return response, extracted_data


# --- Função Central de OCR (Lida com Imagem e PDF) ---
def extract_text_from_file(uploaded_file):
    """
    Processa o arquivo carregado (JPG/PNG ou PDF) e retorna o texto extraído
    usando Tesseract OCR, com PSM 4 para melhor leitura de tabelas.
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)
    
    # Define a configuração para forçar a leitura de colunas e blocos (PSM 4)
    tesseract_config = '--psm 4' 
    
    # 1. Se for PDF
    if "pdf" in file_type:
        st.info("Arquivo PDF detectado. Convertendo primeira página para imagem e extraindo texto...")
        try:
            images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
            if not images:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."
            
            text = pytesseract.image_to_string(images[0], lang='por', config=tesseract_config) 
            st.session_state["image_to_display"] = images[0]
            return text
            
        except Exception as e:
            return f"ERRO_PDF: Verifique se 'poppler-utils' está instalado via packages.txt. Detalhes: {e}"

    # 2. Se for Imagem
    elif "image" in file_type:
        st.info("Arquivo de Imagem detectado. Extraindo texto...")
        try:
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img, lang='por', config=tesseract_config)
            st.session_state["image_to_display"] = img
            return text
        except pytesseract.TesseractNotFoundError:
            return "ERRO_IMAGEM: O Tesseract não está instalado corretamente via packages.txt."
        except Exception as e:
            return f"ERRO_IMAGEM: Falha na extração da imagem. Detalhes: {e}"
            
    return "ERRO_TIPO_INVALIDO: Tipo de arquivo não suportado."


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
    st.error("A chave da API do Gemini (google_api_key) não foi encontrada...")
    st.session_state["llm_ready"] = False


# --- Configuração da Interface Streamlit ---
st.set_page_config(page_title="Extrator Autonometa", layout="wide")
st.title("🤖 Extrator Autonometa (OCR + LLM) de Notas Fiscais")
st.markdown("---")

# --- Seção de Upload e OCR ---
st.header("Upload da Nota Fiscal")

uploaded_file = st.file_uploader(
    "Escolha a Nota Fiscal (JPG, PNG ou PDF):",
    type=['png', 'jpg', 'jpeg', 'pdf']
)

parser = PydanticOutputParser(pydantic_object=NotaFiscal)


if uploaded_file is not None:
    
    # 1. Executa a extração do texto bruto (OCR)
    with st.spinner("Extraindo texto bruto da nota fiscal (OCR)..."):
        ocr_text = extract_text_from_file(uploaded_file)
        
    st.session_state["ocr_text"] = ocr_text

    # 2. Visualização e Verificação
    st.subheader("Visualização e Texto Bruto Extraído")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "image_to_display" in st.session_state:
             st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", use_container_width=True)

    with col2:
        if "ERRO" in ocr_text:
            st.error(f"Erro no OCR: {ocr_text}")
        else:
            st.success("Texto Bruto extraído com sucesso!")
            st.info("Texto (Será enviado ao Gemini para Interpretação):")
            display_text = ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text
            st.code(display_text, language="text")
            
            # Expander para o Texto Bruto Completo para debug
            with st.expander("🔎 Ver Texto Bruto COMPLETO da Nota Fiscal (DEBUG)"):
                 st.code(ocr_text, language="text")

    st.markdown("---")

    # 3. Próxima Etapa: Botão de Interpretação LLM
    if "ERRO" not in ocr_text:
        if st.session_state.get("llm_ready", False):
            if st.button("🚀 Interpretar Dados Estruturados com o Agente Gemini", key="run_extraction_btn"):
                # Limpa o cache para que a nova nota force a extração
                run_llm_extraction_cached.clear() 
                st.session_state["run_llm_extraction"] = True
                st.rerun()
        else:
            st.warning("O Agente Gemini não está pronto. Corrija a API Key para interpretar.")


# --- Seção de Execução da Extração (LLM) ---
if st.session_state.get("run_llm_extraction", False) and st.session_state.get("llm_ready", False):
    
    st.session_state["run_llm_extraction"] = False 
    
    text_to_analyze = st.session_state.get("ocr_text", "")
    response = None # Inicializa 'response' para uso no bloco except
    
    if not text_to_analyze or "ERRO" in text_to_analyze:
        st.error("Não há texto válido para enviar ao Agente LLM.")
        st.stop()

    try:
        # CHAMADA AO LLM AGORA É CACHEADA! Passamos 'parser' como '_parser'
        response, extracted_data = run_llm_extraction_cached(llm, text_to_analyze, parser)
        
        # 4. Exibição dos Resultados
        st.success("✅ Extração concluída com sucesso!")
        
        # Converte o Pydantic object para um dicionário Python simples
        data_dict = extracted_data.model_dump()

        # --- Verificação de Qualidade ---
        quality_warnings = check_for_missing_data(data_dict)
        
        if quality_warnings:
            st.warning("⚠️ Atenção: Diversas informações críticas estão faltando ou ilegíveis na nota fiscal. Isso geralmente ocorre devido à má qualidade da digitalização ou campos não preenchidos.")
            with st.expander("Clique para ver os campos faltantes ou zerados"):
                for warning in quality_warnings:
                    st.markdown(warning)

        st.subheader("Informações Principais")
        
        # --- 4.1 Cabeçalho da Nota com st.columns e st.metric ---
        col_data, col_valor, col_modelo, col_chave = st.columns(4)
        
        col_data.metric("Data de Emissão", data_dict['data_emissao'])
        
        valor_formatado = f"R$ {data_dict['valor_total_nota']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        col_valor.metric("Valor Total da Nota", valor_formatado)
        
        col_modelo.metric("Modelo Fiscal", data_dict['modelo_documento'])
        col_chave.code(data_dict['chave_acesso'])


        # --- 4.2 Detalhes do Emitente e Destinatário com st.expander ---
        st.markdown("---")
        
        with st.expander("🏢 Detalhes do Emitente", expanded=False):
            emitente_data = data_dict.get('emitente', {})
            st.json(emitente_data)

        with st.expander("👤 Detalhes do Destinatário", expanded=False):
            destinatario_data = data_dict.get('destinatario', {})
            st.json(destinatario_data)


        # --- 4.3 Tabela de Itens ---
        st.subheader("🛒 Itens da Nota Fiscal")
        
        itens_list = data_dict.get('itens', [])
        total_tributos_calculado = 0.0

        if itens_list:
            # Calcule o total dos tributos a partir dos itens
            for item in itens_list:
                valor = item.get('valor_aprox_tributos', 0.0)
                if isinstance(valor, (int, float)):
                     total_tributos_calculado += valor

            st.dataframe(
                itens_list,
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
                use_container_width=True
            )
        else:
            st.warning("Nenhum item ou serviço foi encontrado na nota fiscal.")


        # --- 4.4 Exibição dos Totais de Impostos (Com Lógica de Desempate e Edição) ---
        st.markdown("---")
        st.subheader("💰 Totais de Impostos e Despesas")

        impostos_data = data_dict.get('totais_impostos', {})
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

        def formatar_moeda_imp(valor):
            if valor is None or valor == 0.0:
                return "R$ 0,00"
            return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


        col_icms.metric("Base ICMS", formatar_moeda_imp(impostos_data.get('base_calculo_icms')))
        col_icms.metric("Total ICMS", formatar_moeda_imp(impostos_data.get('valor_total_icms')))

        col_ipi.metric("Total IPI", formatar_moeda_imp(impostos_data.get('valor_total_ipi')))

        col_pis.metric("Total PIS", formatar_moeda_imp(impostos_data.get('valor_total_pis')))
        col_cofins.metric("Total COFINS", formatar_moeda_imp(impostos_data.get('valor_total_cofins')))

        col_outras.metric("Outras Despesas", formatar_moeda_imp(impostos_data.get('valor_outras_despesas')))

        col_aprox.metric(f"Total V. Aprox. Tributos{fonte_tributos}", formatar_moeda_imp(total_final_tributos))
        
        
        # --- NOVO BLOCO: Edição Manual Assistida de Impostos Zerados ---
        icms_zerado = impostos_data.get('valor_total_icms', 0.0) <= 0.0
        ipi_zerado = impostos_data.get('valor_total_ipi', 0.0) <= 0.0
        
        if icms_zerado or ipi_zerado:
            st.markdown("---")
            st.subheader("✍️ Edição Manual de Impostos")
            st.info("O Agente LLM não conseguiu extrair os valores detalhados de impostos (ICMS, IPI, PIS/COFINS). Se a nota contém esses valores, insira-os manualmente abaixo para que sejam incluídos no JSON de download.")
            
            icms_val = str(impostos_data.get('valor_total_icms', 0.0))
            ipi_val = str(impostos_data.get('valor_total_ipi', 0.0))
            pis_val = str(impostos_data.get('valor_total_pis', 0.0))
            cofins_val = str(impostos_data.get('valor_total_cofins', 0.0))

            col_edit_icms, col_edit_ipi, col_edit_pis, col_edit_cofins = st.columns(4)
            
            # Usamos 'key' para garantir que o Streamlit salve o estado
            icms_manual = col_edit_icms.text_input("ICMS", value=icms_val, key="manual_icms")
            ipi_manual = col_edit_ipi.text_input("IPI", value=ipi_val, key="manual_ipi")
            pis_manual = col_edit_pis.text_input("PIS", value=pis_val, key="manual_pis")
            cofins_manual = col_edit_cofins.text_input("COFINS", value=cofins_val, key="manual_cofins")
            
            try:
                # Atualiza o data_dict para o download com os valores manuais
                # Substituímos vírgulas por pontos para garantir a conversão float
                data_dict['totais_impostos']['valor_total_icms'] = float(icms_manual.replace(",", "."))
                data_dict['totais_impostos']['valor_total_ipi'] = float(ipi_manual.replace(",", "."))
                data_dict['totais_impostos']['valor_total_pis'] = float(pis_manual.replace(",", "."))
                data_dict['totais_impostos']['valor_total_cofins'] = float(cofins_manual.replace(",", "."))
                st.success("Valores de impostos atualizados para o JSON de download.")
                
            except ValueError:
                st.error("Por favor, insira apenas números válidos (usando ponto ou vírgula) nos campos de edição.")

        # --- Botão de Download ---
        st.markdown("---")
        
        try:
            nome_curto = data_dict['emitente']['nome_razao'].split(' ')[0]
        except (KeyError, IndexError, TypeError):
            nome_curto = "extraida"

        json_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
        st.download_button(
            label="⬇️ Baixar JSON COMPLETO da Extração",
            data=json_data,
            file_name=f"nf_{data_dict['data_emissao']}_{nome_curto}.json",
            mime="application/json"
        )

        with st.expander("Ver JSON Bruto Completo", expanded=False):
             st.json(data_dict)


    except ValidationError as ve:
        st.error("Houve um erro de validação (Pydantic). O Gemini pode ter retornado um JSON malformado.")
        # Exibe o JSON que o Gemini tentou retornar para debug
        if response is not None:
            with st.expander("Ver Resposta Bruta do LLM (JSON malformado)", expanded=True):
                st.code(response.content, language='json')
        st.warning(f"Detalhes do Erro: {ve}")

    except Exception as e:
        # TRATAMENTO DE ERRO GENÉRICO
        st.error(f"Houve um erro geral durante a interpretação pelo Gemini. Detalhes: {e}")
        
        if 'response' in locals() and response is not None:
            with st.expander("Ver Resposta Bruta do LLM (Debugging)", expanded=True):
                st.code(response.content, language='json')
        
        st.warning("O Agente LLM pode ter falhado ao processar o texto OCR.")

st.markdown("---")
