# extrator.py

import streamlit as st
from PIL import Image
import pytesseract         # OCR
from pdf2image import convert_from_bytes
from io import BytesIO
import json                # NECESSÁRIO para st.download_button e JSON dumps

# --- Imports LangChain e Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

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
        validation_alias='nome_raza' # Aceita 'nome_raza' na entrada JSON
    )
    endereco_completo: str = Field(description="Endereço completo (Rua, Número, Bairro, Cidade, Estado).")
    inscricao_estadual: str = Field(description="Inscrição Estadual, se disponível.")

# Sub-estrutura para os Totais de Impostos (Nível de Nota)
class TotaisImposto(BaseModel):
    base_calculo_icms: float = Field(description="Valor total da Base de Cálculo do ICMS da nota.")
    valor_total_icms: float = Field(description="Valor total do ICMS destacado na nota.")
    valor_total_ipi: float = Field(description="Valor total do IPI destacado na nota.")
    valor_total_pis: float = Field(description="Valor total do PIS destacado na nota.")
    valor_total_cofins: float = Field(description="Valor total do COFINS destacado na nota.")
    valor_aprox_tributos: float = Field(description="Valor aproximado total dos tributos (Lei da Transparência). Este campo deve ser preenchido se o valor for encontrado no bloco de Dados Adicionais ou como um total único.") 
    valor_outras_despesas: float = Field(description="Valor total de outras despesas acessórias (frete, seguro, etc.).")
    

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

    # TOTAIS DE IMPOSTOS
    totais_impostos: TotaisImposto = Field(description="Valores totais de impostos e despesas acessórias da nota.")
    
    # Itens/Serviços (Lista)
    itens: list[ItemNota] = Field(description="Lista completa de todos os produtos ou serviços discriminados na nota, seguindo o esquema ItemNota.")

def check_for_missing_data(data_dict: dict) -> list:
    """Verifica se há dados críticos faltantes ou zerados e retorna uma lista de avisos."""
    warnings = []
    
    # 1. Campos principais obrigatórios (Strings)
    critical_str_fields = {
        'Chave de Acesso': data_dict.get('chave_acesso', ''),
        'Data de Emissão': data_dict.get('data_emissao', ''),
    }
    
    # 2. Checagem de Emitente/Destinatário
    emitente = data_dict.get('emitente', {})
    destinatario = data_dict.get('destinatario', {})

    critical_str_fields['CNPJ/CPF do Emitente'] = emitente.get('cnpj_cpf', '')
    critical_fields_emitente = {
        'Nome/Razão do Emitente': emitente.get('nome_razao', '')
    }
    
    critical_str_fields['CNPJ/CPF do Destinatário'] = destinatario.get('cnpj_cpf', '')
    critical_fields_destinatario = {
        'Nome/Razão do Destinatário': destinatario.get('nome_razao', '')
    }

    # Checa campos strings ausentes/zerados
    all_str_fields = {**critical_str_fields, **critical_fields_emitente, **critical_fields_destinatario}
    for name, value in all_str_fields.items():
        # Considera 'vazio' se for string vazia, None, ou '0'/'0.0'
        if not value or value.strip() == '0' or value.strip() == '0.0':
            warnings.append(f"❌ O campo '{name}' está vazio ou ilegível.")

    # 3. Checagem de Valores (Floats)
    valor_total_nota = data_dict.get('valor_total_nota', 0.0)
    if valor_total_nota <= 0.0:
        warnings.append("❌ O 'Valor Total da Nota' está zerado (R$ 0,00).")
    
    # 4. Checagem da lista de itens
    if not data_dict.get('itens'):
        warnings.append("❌ A lista de Itens/Produtos está vazia.")
    
    return warnings

# --- Função Central de OCR (Lida com Imagem e PDF) ---
def extract_text_from_file(uploaded_file):
    """
    Processa o arquivo carregado (JPG/PNG ou PDF) e retorna o texto extraído
    usando Tesseract OCR.
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)
    
    tesseract_config = '--psm 4' 
    
    # 1. Se for PDF
    if "pdf" in file_type:
        st.info("Arquivo PDF detectado. Convertendo primeira página para imagem e extraindo texto...")
        try:
            # Converte a primeira página do PDF para uma imagem PIL em memória
            images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
            if not images:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."
            
            # Executa OCR na imagem convertida com o novo PSM
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
            # Executa OCR diretamente na imagem com o novo PSM
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
        # Exibe a imagem (ou a imagem da primeira página do PDF)
        if "image_to_display" in st.session_state:
             st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", width='stretch')

    with col2:
        if "ERRO" in ocr_text:
            st.error(f"Erro no OCR: {ocr_text}")
        else:
            st.success("Texto Bruto extraído com sucesso!")
            st.info("Texto (Será enviado ao Gemini para Interpretação):")
            # Limita a visualização para não poluir a tela
            display_text = ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text
            st.code(display_text, language="text")
            
    with st.expander("🔎 Ver Texto Bruto COMPLETO da Nota Fiscal (DEBUG)"):
        st.code(ocr_text, language="text") # Mostra o texto completo, sem limite de 1000 caracteres
        
    st.markdown("---")

    # 3. Próxima Etapa: Botão de Interpretação LLM
    if "ERRO" not in ocr_text:
        if st.session_state.get("llm_ready", False):
            if st.button("🚀 Interpretar Dados Estruturados com o Agente Gemini", key="run_extraction_btn"):
                st.session_state["run_llm_extraction"] = True
                st.rerun() # CORRIGIDO: st.experimental_rerun() -> st.rerun()
        else:
            st.warning("O Agente Gemini não está pronto. Corrija a API Key para interpretar.")


# --- Seção de Execução da Extração (LLM) ---
if st.session_state.get("run_llm_extraction", False) and st.session_state.get("llm_ready", False):
    
    st.session_state["run_llm_extraction"] = False 
    
    # 1. Recupera o texto bruto do OCR e a resposta bruta do LLM
    text_to_analyze = st.session_state.get("ocr_text", "")
    response = None # Inicializa 'response' para uso no bloco except
    
    if not text_to_analyze or "ERRO" in text_to_analyze:
        st.error("Não há texto válido para enviar ao Agente LLM.")
        st.stop()

    with st.spinner("O Agente Gemini está interpretando o texto para extrair dados estruturados..."):
        try:
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

            # 2b. Preenche o template com as instruções do parser (CRUCIAL para evitar erro de variáveis)
            prompt_values = prompt_template.partial(
                format_instructions=parser.get_format_instructions()
            )
            
            final_prompt = prompt_values.format_messages(text_to_analyze=text_to_analyze)

            # 3. Execução do LLM e Parsin'
            response = llm.invoke(final_prompt) # Armazena a resposta bruta
            extracted_data: NotaFiscal = parser.parse(response.content) # Tenta o parsing no Pydantic

            # 4. Exibição dos Resultados
            st.success("✅ Extração concluída com sucesso!")
            
            # Converte o Pydantic object para um dicionário Python simples
            data_dict = extracted_data.model_dump()

            # Valiação da qualiiade e conteúdo da digitalização da NF
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
            
            # Formatando valor_total_nota para moeda brasileira
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
                # 1. Calcule o total dos tributos a partir dos itens
                for item in itens_list:
                    valor = item.get('valor_aprox_tributos', 0.0)
                    if isinstance(valor, (int, float)):
                         total_tributos_calculado += valor
            
                # 2. Exibe o DataFrame (Mantido igual com a nova coluna)
                st.dataframe(
                    itens_list,
                    column_order=[
                        "descricao", "quantidade", "valor_unitario", "valor_total", 
                        "codigo_cfop", "cst_csosn", "valor_aprox_tributos"
                    ],
                    column_config={
                        # ... (configuração das colunas, mantida igual) ...
                        "valor_aprox_tributos": st.column_config.NumberColumn("V. Aprox. Tributos", format="R$ %.2f")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("Nenhum item ou serviço foi encontrado na nota fiscal.")
            
            
            # --- NOVO BLOCO: Exibição dos Totais de Impostos (Com Lógica de Desempate) ---
            st.markdown("---")
            st.subheader("💰 Totais de Impostos e Despesas")
            
            impostos_data = data_dict.get('totais_impostos', {})
            total_tributos_extraido_direto = impostos_data.get('valor_aprox_tributos', 0.0)
            
            # LÓGICA DE DESEMPATE CRÍTICA:
            if total_tributos_calculado > 0.0:
                # Cenário A: Prioriza o valor calculado (se o LLM extraiu por item)
                total_final_tributos = total_tributos_calculado
                fonte_tributos = " (Calculado dos Itens)"
            elif total_tributos_extraido_direto > 0.0:
                # Cenário B: Usa o valor extraído diretamente (se veio dos Dados Adicionais)
                total_final_tributos = total_tributos_extraido_direto
                fonte_tributos = " (Extraído dos Dados Adicionais)"
            else:
                # Cenário C: Não encontrado
                total_final_tributos = 0.0
                fonte_tributos = ""
            
            
            col_icms, col_ipi, col_pis, col_cofins, col_outras, col_aprox = st.columns(6)
            
            # Função auxiliar para formatar moeda (mantida igual)
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
            
            # NOVO: Exibe o Total Aproximado COM a fonte da informação
            col_aprox.metric(f"Total V. Aprox. Tributos{fonte_tributos}", formatar_moeda_imp(total_final_tributos))

            # --- 4.4 Botão de Download ---
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


        except Exception as e:
            # --- TRATAMENTO DE ERRO MELHORADO ---
            st.error(f"Houve um erro durante a interpretação pelo Gemini. Detalhes: {e}")
            
            # Se a falha foi no Pydantic, exibe a resposta bruta do LLM para debug
            if response is not None:
                with st.expander("Ver Resposta Bruta do LLM (JSON malformado)", expanded=True):
                    st.code(response.content, language='json')
            
            st.warning("O Agente LLM pode ter falhado ao extrair a estrutura JSON a partir do texto OCR.")

st.markdown("---")
