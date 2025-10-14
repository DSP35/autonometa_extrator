### Autonometa
### Extator de dados de Notas Fiscais
### Desenvolvido por David Parede

import streamlit as st
from PIL import Image

# --- Imports LangChain e Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain 
import os 

# --- Configuração do Modelo Gemini ---
if "google_api_key" in st.secrets:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  
            google_api_key=st.secrets["google_api_key"]
        )
        st.session_state["llm_ready"] = True # Sinaliza que o LLM está pronto
        
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo Gemini. Detalhes: {e}")
        st.session_state["llm_ready"] = False
        llm = None
else:
    st.error(
        "A chave da API do Gemini (google_api_key) não foi encontrada nos secrets do Streamlit. "
        "Por favor, configure-a no arquivo .streamlit/secrets.toml ou no painel do Cloud."
    )
    st.session_state["llm_ready"] = False
    llm = None

# --- Configuração da Interface Streamlit ---
st.set_page_config(page_title="Extrator Autonometa", layout="wide")
st.title("🤖 Extrator Autonometa de Notas Fiscais")
st.markdown("---")

# --- Seção de Teste do Agente LLM ---
st.subheader("Teste de Conexão com o Agente LLM")

if st.session_state.get("llm_ready", False):
    st.success("✅ Agente LLM (Gemini) inicializado com sucesso!")

    # Botão para testar a comunicação com o LLM
    if st.button("Testar Agente"):
        with st.spinner("Enviando pergunta para o agente..."):
            try:
                # Pergunta simples para o teste
                prompt = ChatPromptTemplate.from_messages([
                    ("human", "Responda de forma curta e direta: Qual o seu propósito?")
                ])
                chain = prompt | llm
                response = chain.invoke({})

                # Exibe a resposta
                st.info(f"**Resposta do Agente:** {response.content}")

            except Exception as e:
                st.error(f"Ocorreu um erro ao se comunicar com o agente: {e}")
else:
    st.warning("O agente LLM não está pronto. Verifique a configuração da API Key.")

st.markdown("---")

# Espaço reservado para as próximas funcionalidades
st.header("Próximos Passos")
st.info("Aqui implementaremos o upload da nota fiscal e a extração dos dados.")
