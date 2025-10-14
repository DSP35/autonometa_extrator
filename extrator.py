### Autonometa
### Extator de dados de Notas Fiscais
### Desenvolvido por David Parede

# extrator.py

import streamlit as st
from PIL import Image
import pytesseract         # OCR
from pdf2image import convert_from_bytes # Novo: Para converter PDF em Imagem
from io import BytesIO

# --- Imports LangChain e Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# --- 1. Definindo o Schema de Saída ---
class NotaFiscal(BaseModel):
    """Estrutura Padrão dos Dados de uma Nota Fiscal."""
    cnpj_emitente: str = Field(description="CNPJ da empresa que emitiu a nota fiscal (apenas dígitos).")
    nome_emitente: str = Field(description="Nome ou razão social do emitente da nota fiscal.")
    data_emissao: str = Field(description="Data de emissão da nota fiscal no formato YYYY-MM-DD.")
    valor_total: float = Field(description="Valor total da nota fiscal.")
    itens_servicos: list[str] = Field(description="Lista com a descrição dos principais produtos ou serviços na nota. Retorne no máximo 3 itens.")

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
             st.image(st.session_state["image_to_display"], caption="Nota Fiscal Processada", use_container_width=True)

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
            # 2. Criando o Prompt de Extração de Texto
            format_instructions = parser.get_format_instructions()
            
            full_human_prompt = (
                "Analise o texto a seguir e extraia os campos fiscais na estrutura JSON. "
                "Se o valor for um texto/string, use aspas. Para valores numéricos, use float.\n\n"
                f"INSTRUÇÕES DE FORMATO:\n{format_instructions}\n\n"
                f"TEXTO BRUTO DA NOTA: \n{text_to_analyze}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "Você é um agente de extração de dados fiscais. Sua tarefa é analisar o texto bruto fornecido de uma nota fiscal e extrair as informações solicitadas no formato JSON. Seja rigoroso com o formato e não invente dados."),
                    ("human", full_human_prompt),
                ]
            )

            # 3. Execução da Chain (LLM + Parser)
            chain = prompt_template | llm | parser
            extracted_data: NotaFiscal = chain.invoke({})

            # 4. Exibição dos Resultados
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
