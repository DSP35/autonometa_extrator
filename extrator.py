### Autonometa
### Extator de dados de Notas Fiscais
### Desenvolvido por David Parede

# extrator.py

import streamlit as st
from PIL import Image
import pytesseract 
from pdf2image import convert_from_bytes 
from io import BytesIO
import json

# --- Imports LangChain e Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. Definindo o Schema de Saída (Estrutura da Nota Fiscal) ---
# Sub-estrutura para cada Item da Nota
class ItemNota(BaseModel):
    descricao: str = Field(description="Nome ou descrição completa do produto/serviço.")
    quantidade: float = Field(description="Quantidade do item, convertida para um valor numérico (float).")
    valor_unitario: float = Field(description="Valor unitário do item.")
    valor_total: float = Field(description="Valor total da linha do item.")
    codigo_cfop: str = Field(description="Código CFOP (Natureza da Operação) associado ao item, se disponível.")
    cst_csosn: str = Field(description="Código CST (Situação Tributária) ou CSOSN do item, se disponível.")
    icms_valor: float = Field(description="Valor do ICMS incidido sobre o item, se disponível.")

# Sub-estrutura para Emitente e Destinatário
class ParteFiscal(BaseModel):
    cnpj_cpf: str = Field(description="CNPJ ou CPF da parte fiscal (apenas dígitos).")
    nome_razao: str = Field(description="Nome ou Razão Social completa.")
    endereco_completo: str = Field(description="Endereço completo (Rua, Número, Bairro, Cidade, Estado).")
    inscricao_estadual: str = Field(description="Inscrição Estadual, se disponível.")

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
    
    # Itens/Serviços (Lista)
    itens: list[ItemNota] = Field(description="Lista completa de todos os produtos ou serviços discriminados na nota, seguindo o esquema ItemNota.")

# --- Função Central de OCR (Lida com Imagem e PDF) ---
def extract_text_from_file(uploaded_file):
    """
    Processa o arquivo carregado (JPG/PNG ou PDF) e retorna o texto extraído
    usando Tesseract OCR.
    """
    file_type = uploaded_file.type
    uploaded_file.seek(0)
    
    # 1. Se for PDF
    if "pdf" in file_type:
        st.info("Arquivo PDF detectado. Convertendo primeira página para imagem e extraindo texto...")
        try:
            # Converte a primeira página do PDF para uma imagem PIL em memória
            images = convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
            if not images:
                return "ERRO_CONVERSAO: Não foi possível converter o PDF em imagem."
            
            # Executa OCR na imagem convertida (primeira página)
            text = pytesseract.image_to_string(images[0], lang='por') 
            st.session_state["image_to_display"] = images[0] # Salva a imagem para visualização
            return text
            
        except Exception as e:
            return f"ERRO_PDF: Verifique se 'poppler-utils' está instalado via packages.txt. Detalhes: {e}"

    # 2. Se for Imagem
    elif "image" in file_type:
        st.info("Arquivo de Imagem detectado. Extraindo texto...")
        try:
            img = Image.open(uploaded_file)
            # Executa OCR diretamente na imagem
            text = pytesseract.image_to_string(img, lang='por')
            st.session_state["image_to_display"] = img # Salva a imagem para visualização
            return text
        except Exception as e:
            return f"ERRO_IMAGEM: Verifique se 'tesseract-ocr' está instalado via packages.txt. Detalhes: {e}"
            
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

    st.markdown("---")

    # 3. Próxima Etapa: Botão de Interpretação LLM
    if "ERRO" not in ocr_text:
        if st.session_state.get("llm_ready", False):
            if st.button("🚀 Interpretar Dados Estruturados com o Agente Gemini", key="run_extraction_btn"):
                st.session_state["run_llm_extraction"] = True
                st.rerun()
        else:
            st.warning("O Agente Gemini não está pronto. Corrija a API Key para interpretar.")


# --- Seção de Execução da Extração (LLM) ---
if st.session_state.get("run_llm_extraction", False) and st.session_state.get("llm_ready", False):
    
    st.session_state["run_llm_extraction"] = False 
    st.subheader("Análise e Extração Estruturada de Dados 🧠")
    
    # 1. Recupera o texto bruto do OCR
    text_to_analyze = st.session_state.get("ocr_text", "")

    if not text_to_analyze or "ERRO" in text_to_analyze:
        st.error("Não há texto válido para enviar ao Agente LLM.")
        st.stop()

    with st.spinner("O Agente Gemini está interpretando o texto para extrair dados estruturados..."):
        try:
            # 2. Criando o Prompt de Extração de Texto com PromptTemplate simples

            # 2a. Define o template com UMA variável de entrada (text_to_analyze)
            # Todo o resto (instruções e sistema) é tratado como strings literais.
            # Usamos o {format_instructions} e o {text_to_analyze} como placeholders.

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    # REFORÇANDO A INSTRUÇÃO DE SISTEMA:
                    ("system", "Você é um agente de extração de dados fiscais. Sua tarefa é analisar o texto bruto fornecido de uma nota fiscal e extrair TODAS as informações solicitadas no formato JSON. ATENÇÃO ESPECIAL: Você deve extrair todas as listas e sub-objetos (EMITENTE, DESTINATÁRIO, e a LISTA DE ITENS) de forma completa e exata. Não invente dados."),
                    
                    ("human", (
                        "Analise o texto a seguir e extraia os campos fiscais na estrutura JSON. "
                        "Converta todos os valores de impostos, totais e quantidades para o tipo float. "
                        "Forneça a lista completa de itens, mesmo que confusa. \n\n"
                        "INSTRUÇÕES DE FORMATO:\n"
                        "{format_instructions}\n\n"
                        "TEXTO BRUTO DA NOTA:\n"
                        "{text_to_analyze}"
                    )),
                ]
            )

            # 2b. Combinamos a saída do parser e o texto da nota com o template
            # Esta é a etapa CRUCIAL: 'format_instructions' e 'text_to_analyze' são passados como variáveis
            # A LangChain não tenta analisar o conteúdo interno do format_instructions.
            
            prompt_values = prompt_template.partial(
                format_instructions=parser.get_format_instructions()
            )
            
            final_prompt = prompt_values.format_messages(text_to_analyze=text_to_analyze)


            # 3. Execução da Chain
            # A chain agora é mais simples, pois o parser só precisa verificar a saída do LLM.
            
            response = llm.invoke(final_prompt)
            
            # Usamos o parser para garantir que a saída do LLM seja validada no Pydantic
            extracted_data: NotaFiscal = parser.parse(response.content)


            # 4. Exibição dos Resultados (o bloco try/except e o restante do código permanecem os mesmos)
            st.success("✅ Extração concluída com sucesso!")
            
            data_dict = extracted_data.model_dump()
            
            st.dataframe(
                data={
                    "Campo": list(data_dict.keys()),
                    "Valor Extraído": list(data_dict.values())
                },
                use_container_width=True
            )
            
            # Adicionar a funcionalidade de Download (Próxima etapa)
            json_data = json.dumps(data_dict, ensure_ascii=False, indent=4)
            st.download_button(
                label="⬇️ Baixar JSON da Extração",
                data=json_data,
                file_name="nota_fiscal_extraida.json",
                mime="application/json"
            )

            with st.expander("Ver JSON Bruto"):
                 st.json(data_dict)

        except Exception as e:
            st.error(f"Houve um erro durante a interpretação pelo Gemini. Detalhes: {e}")
            st.warning("O Agente LLM pode ter falhado ao extrair a estrutura JSON a partir do texto OCR.")
            

st.markdown("---")
