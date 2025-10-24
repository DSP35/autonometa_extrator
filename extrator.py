# Extrator Autonometa
# Desenvolvido por David Parede

import os, re, io, sys, json, tempfile, logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("autonometa.extrator")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

class NotaItem(BaseModel):
    descricao: str = Field("", description="Descrição do item")
    ncm: Optional[str] = None
    cfop: Optional[str] = None
    quantidade: Optional[float] = None
    valor_unitario: Optional[float] = None
    valor_total: Optional[float] = None

class NotaFiscal(BaseModel):
    chave_acesso: Optional[str] = None
    emitente: Dict[str, Any] = Field(default_factory=dict)
    destinatario: Dict[str, Any] = Field(default_factory=dict)
    itens: List[NotaItem] = Field(default_factory=list)
    impostos: Dict[str, Any] = Field(default_factory=dict)
    metadados: Dict[str, Any] = Field(default_factory=dict)

import numpy as _np
def _variance_of_laplacian(gray):
    import cv2
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def assess_image_quality(image_bgr):
    import cv2, numpy as np
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharp = _variance_of_laplacian(gray)
    contrast = float(np.std(gray))
    score = 0.0
    if w * h >= 1024 * 768: score += 0.4
    if sharp >= 100.0: score += 0.4
    if contrast >= 30.0: score += 0.2
    return {"width": int(w), "height": int(h), "sharpness": float(sharp), "contrast": float(contrast), "quality_score": float(score)}

def enhance_image(image_bgr):
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    eq = cv2.equalizeHist(den)
    gauss = cv2.GaussianBlur(eq, (0, 0), 3)
    usm = cv2.addWeighted(eq, 1.5, gauss, -0.5, 0)
    th = cv2.adaptiveThreshold(usm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def load_image(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Não foi possível carregar a imagem para OCR.")
    return img

def pdf_to_images_all_pages(pdf_path):
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=300)
    if not pages:
        raise ValueError("PDF sem páginas legíveis.")
    paths = []
    for i, pg in enumerate(pages):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".p{i+1}.png")
        pg.save(tmp.name, "PNG")
        paths.append(tmp.name)
    return paths

def parse_nfe_xml(xml_bytes):
    from lxml import etree
    def _text(x): return x.text.strip() if x is not None and x.text else None
    root = etree.fromstring(xml_bytes)
    nsmap = root.nsmap.copy()
    if None in nsmap: nsmap["n"] = nsmap.pop(None)
    def find(path):
        el = root.find(path, namespaces=nsmap)
        if el is None and path.startswith(".//"):
            el = root.find(path.replace(".//",""), namespaces=nsmap)
        return el
    def finds(path):
        els = root.findall(path, namespaces=nsmap)
        if not els and path.startswith(".//"):
            els = root.findall(path.replace(".//",""), namespaces=nsmap)
        return els

    icmstot = find(".//n:ICMSTot") or find(".//ICMSTot")
    emitente = {
        "nome": _text(find(".//n:emit/n:xNome")) or _text(find(".//emit/xNome")),
        "cnpj": _text(find(".//n:emit/n:CNPJ")) or _text(find(".//emit/CNPJ")),
        "ie": _text(find(".//n:emit/n:IE")) or _text(find(".//emit/IE")),
        "uf": _text(find(".//n:emit/n:enderEmit/n:UF")) or _text(find(".//emit/enderEmit/UF")),
    }
    destinatario = {
        "nome": _text(find(".//n:dest/n:xNome")) or _text(find(".//dest/xNome")),
        "cnpj": _text(find(".//n:dest/n:CNPJ")) or _text(find(".//dest/CNPJ")) or _text(find(".//n:dest/n:CPF")) or _text(find(".//dest/CPF")),
        "ie": _text(find(".//n:dest/n:IE")) or _text(find(".//dest/IE")),
        "uf": _text(find(".//n:dest/n:enderDest/n:UF")) or _text(find(".//dest/enderDest/UF")),
    }

    itens = []
    for det in finds(".//n:det") + finds(".//det"):
        prod = det.find(".//n:prod", namespaces=nsmap) or det.find(".//prod")
        if prod is None: continue
        def fnum(v):
            if v is None: return None
            v = v.replace(",", ".")
            try: return float(v)
            except: return None
        item = {
            "descricao": _text(prod.find("n:xProd", namespaces=nsmap)) or _text(prod.find("xProd")) or "",
            "ncm": _text(prod.find("n:NCM", namespaces=nsmap)) or _text(prod.find("NCM")),
            "cfop": _text(prod.find("n:CFOP", namespaces=nsmap)) or _text(prod.find("CFOP")),
            "quantidade": fnum(_text(prod.find("n:qCom", namespaces=nsmap)) or _text(prod.find("qCom"))),
            "valor_unitario": fnum(_text(prod.find("n:vUnCom", namespaces=nsmap)) or _text(prod.find("vUnCom"))),
            "valor_total": fnum(_text(prod.find("n:vProd", namespaces=nsmap)) or _text(prod.find("vProd"))),
        }
        itens.append(item)

    def fnum(v):
        if v is None: return None
        v = v.replace(",", ".")
        try: return float(v)
        except: return None

    impostos = {
        "icms": fnum(_text(icmstot.find("n:vICMS", namespaces=nsmap)) if icmstot is not None else None),
        "ipi": fnum(_text(icmstot.find("n:vIPI", namespaces=nsmap)) if icmstot is not None else None),
        "pis": fnum(_text(icmstot.find("n:vPIS", namespaces=nsmap)) if icmstot is not None else None),
        "cofins": fnum(_text(icmstot.find("n:vCOFINS", namespaces=nsmap)) if icmstot is not None else None),
    }
    total_nf = fnum(_text(icmstot.find("n:vNF", namespaces=nsmap)) if icmstot is not None else None)
    data = {
        "chave_acesso": None,
        "emitente": emitente,
        "destinatario": destinatario,
        "itens": itens,
        "impostos": impostos,
        "metadados": {"layout": "nfe_xml", "total_nf": total_nf}
    }
    return data

def run_ocr(image_path):
    try:
        import pytesseract
        from PIL import Image as PILImage
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        return pytesseract.image_to_string(PILImage.open(image_path), lang="por+eng")
    except Exception as e:
        logger.exception("Falha no OCR: %s", e); return ""

PROMPT_TEMPLATE = """
Você é um extrator de dados fiscais. A partir do TEXTO OCR abaixo, devolva um JSON válido com os campos:
- chave_acesso (string ou null, 44 dígitos se existir)
- emitente: {"nome": ..., "cnpj": ...}
- destinatario: {"nome": ..., "cnpj": ...}
- itens: lista de itens com campos: descricao, ncm, cfop, quantidade, valor_unitario, valor_total
- impostos: {"icms":..., "ipi":..., "pis":..., "cofins":...}
- metadados: {"layout": "<nfe_texto|nfce_texto|nfe_chave_44|desconhecido>"}
Responda apenas o JSON, sem comentários.
TEXTO OCR:
{texto}
"""

def parse_with_llm(text):
    if not GOOGLE_API_KEY: return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | model
        resp = chain.invoke({"texto": text})
        content = getattr(resp, "content", str(resp)).strip()
        if content.startswith("```"):
            import re
            content = re.sub(r"^```.*?\n", "", content, flags=re.S).rstrip("`").rstrip()
        import json, re
        try: return json.loads(content)
        except: 
            m = re.search(r"\{.*\}", content, re.S)
            return json.loads(m.group(0)) if m else None
    except Exception as e:
        logger.warning("LLM indisponível (%s). Usando heurística.", e); return None

def detect_layout(text):
    import re
    t = (text or "").lower()
    if "chave de acesso" in t or "nfe" in t: return "nfe_texto"
    if "nfce" in t or "nota fiscal eletrônica" in t: return "nfce_texto"
    if re.search(r"\b\d{44}\b", t): return "nfe_chave_44"
    return "desconhecido"

def extract_tax_codes(text):
    import re
    text = text or ""
    cfops = re.findall(r"\b([1-7]\d{3})\b", text)
    cst = [m.group(1) for m in re.finditer(r"(?:CST[:\s]*)(\d{2,3})", text, re.I)]
    ncm = re.findall(r"\b(\d{8})\b", text)
    def dedup(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen: seen.add(x); out.append(x)
        return out
    return {"cfop": dedup(cfops), "cst": dedup(cst), "ncm": dedup(ncm)}

def parse_with_heuristics(text):
    import re
    layout = detect_layout(text)
    data = {"chave_acesso": None, "emitente": {"nome": None, "cnpj": None}, "destinatario": {"nome": None, "cnpj": None},
            "itens": [], "impostos": {}, "metadados": {"layout": layout}}
    m = re.search(r"\b\d{44}\b", text or "")
    if m: data["chave_acesso"] = m.group(0)
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for ln in lines[:120]:
        if any(k in ln.lower() for k in ["qtd","quantidade"]):
            data["itens"].append({"descricao": ln, "ncm": None, "cfop": None, "quantidade": None, "valor_unitario": None, "valor_total": None})
    return data

def parse_text(text):
    llm = parse_with_llm(text)
    base = llm if llm is not None else parse_with_heuristics(text)
    codes = extract_tax_codes(text)
    base.setdefault("impostos", {})
    base["impostos"].update({k: v for k, v in codes.items() if v})
    return base

def validate_to_model(data):
    warnings = []
    try:
        if "itens" in data and isinstance(data["itens"], list):
            norm = []
            for i in data["itens"]:
                if isinstance(i, dict): norm.append(NotaItem(**i).dict())
                elif isinstance(i, NotaItem): norm.append(i.dict())
            data["itens"] = norm
        nf = NotaFiscal(**data)
        if not nf.emitente.get("nome"): warnings.append("emitente.nome ausente")
        if not nf.destinatario.get("nome"): warnings.append("destinatario.nome ausente")
        return nf, warnings
    except ValidationError as e:
        warnings.append(f"Falha de validação: {e}")
        return NotaFiscal(), warnings

def process_document(path):
    # XML
    if path.lower().endswith((".xml",".nfe",".nfce",".cte",".mdfe")):
        with open(path, "rb") as f: xml_bytes = f.read()
        parsed = parse_nfe_xml(xml_bytes)
        nf, warns = validate_to_model(parsed)
        qm = {"width":0,"height":0,"sharpness":0,"contrast":0,"quality_score":1.0}
        return {"nota_fiscal": nf.dict(), "qualidade_media": qm, "paginas": 1, "melhorada_em_quais": [], "avisos": warns, "instrucoes_qualidade": []}

    # PDF ou Imagem
    pages = [path]
    if path.lower().endswith(".pdf"):
        pages = pdf_to_images_all_pages(path)

    import cv2, tempfile
    qualities, improved_flags, texts = [], [], []
    for p in pages:
        img = load_image(p)
        q = assess_image_quality(img); qualities.append(q)
        if q["quality_score"] < 0.7:
            img = enhance_image(img); improved_flags.append(True)
        else:
            improved_flags.append(False)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(tmp_img.name, img)
        texts.append(run_ocr(tmp_img.name))

    merged_text = "\n\n".join(texts)
    parsed = parse_text(merged_text)
    parsed.setdefault("metadados", {})
    parsed["metadados"]["layout"] = parsed["metadados"].get("layout") or detect_layout(merged_text)
    parsed["metadados"]["ocr_chars"] = len(merged_text); parsed["metadados"]["pages"] = len(pages)

    nf, warns = validate_to_model(parsed)
    if qualities:
        qm = {k: sum(q[k] for q in qualities)/len(qualities) for k in ["width","height","sharpness","contrast","quality_score"]}
    else:
        qm = {"width":0,"height":0,"sharpness":0,"contrast":0,"quality_score":0}

    return {"nota_fiscal": nf.dict(), "qualidade_media": qm, "paginas": len(pages),
            "melhorada_em_quais": improved_flags, "avisos": warns, "instrucoes_qualidade": quality_guidance(qm)}

def quality_guidance(q):
    tips = []
    if q.get("quality_score", 0) < 0.7:
        tips.append("A imagem está com qualidade baixa. Ilumine melhor e evite sombras.")
        tips.append("Apoie o celular, centralize a nota e mantenha a câmera paralela ao documento.")
        tips.append("Use resolução adequada (mín. ~1024x768) ou 300 DPI em PDF.")
    return tips

def _safe_float(x):
    try: return float(x)
    except: return None

def audit_nota(nf):
    issues = []
    emit = nf.get("emitente", {}) or {}
    dest = nf.get("destinatario", {}) or {}
    itens = nf.get("itens", []) or []
    imp = nf.get("impostos", {}) or {}
    meta = nf.get("metadados", {}) or {}

    if not emit.get("nome"): issues.append("Emitente sem nome.")
    if not dest.get("nome"): issues.append("Destinatário sem nome.")

    cfops = set(); ncms = set()
    for i, it in enumerate(itens, 1):
        if not it.get("descricao"): issues.append(f"Item {i} sem descrição.")
        if it.get("cfop"): cfops.add(str(it.get("cfop")))
        if it.get("ncm"): ncms.add(str(it.get("ncm")))
    if not cfops: issues.append("Nenhum CFOP identificado.")
    if not ncms: issues.append("Nenhum NCM identificado.")

    soma_itens = sum([_safe_float(i.get("valor_total")) or 0 for i in itens])
    total_nf = _safe_float(meta.get("total_nf"))
    if total_nf is not None and abs(soma_itens - total_nf) > 0.05:
        issues.append(f"Soma de itens ({soma_itens:.2f}) diverge do total da NF ({total_nf:.2f}).")

    for k in ["icms","ipi","pis","cofins"]:
        v = imp.get(k)
        if v is not None:
            try: float(v)
            except: issues.append(f"Imposto '{k}' com formato inválido: {v}")

    return {"inconsistencias": issues, "resumo": {"qtd_itens": len(itens), "soma_itens": round(soma_itens,2),
            "total_nf": total_nf, "cfops": sorted(list(cfops)), "ncms": sorted(list(ncms))}}

def aggregate_results(results):
    total_notas = len(results); total_valor = 0.0; total_itens = 0
    total_icms = total_ipi = total_pis = total_cofins = 0.0; problemas = 0
    for r in results:
        nf = r.get("nota_fiscal", {})
        meta = nf.get("metadados", {}) or {}
        imp = nf.get("impostos", {}) or {}
        total_itens += len(nf.get("itens", []) or [])
        try: total_valor += float(meta.get("total_nf") or 0)
        except: pass
        for k in ["icms","ipi","pis","cofins"]:
            try:
                v = float(imp.get(k) or 0)
                if k=='icms': total_icms+=v
                elif k=='ipi': total_ipi+=v
                elif k=='pis': total_pis+=v
                elif k=='cofins': total_cofins+=v
            except: pass
        rep = audit_nota(nf)
        if rep["inconsistencias"]: problemas += 1
    return {"total_documentos": total_notas, "total_itens": total_itens, "valor_total": round(total_valor,2),
            "impostos_total": {"icms": round(total_icms,2), "ipi": round(total_ipi,2), "pis": round(total_pis,2), "cofins": round(total_cofins,2)},
            "documentos_com_problemas": problemas}

def export_pdf_resumo(nota, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    c = canvas.Canvas(path, pagesize=A4); w, h = A4; y = h - 2*cm
    c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, "Resumo Fiscal — AutonoMeta"); y -= 1*cm
    c.setFont("Helvetica", 10)
    def line(txt):
        nonlocal y; c.drawString(2*cm, y, (txt or "")[:110]); y -= 0.6*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm
    line(f"Emitente: {nota.get('emitente',{}).get('nome')}  CNPJ: {nota.get('emitente',{}).get('cnpj')}")
    line(f"Destinatário: {nota.get('destinatario',{}).get('nome')}  CNPJ/CPF: {nota.get('destinatario',{}).get('cnpj')}")
    mt = nota.get('metadados',{})
    line(f"Itens: {len(nota.get('itens', []))}  Total NF: {mt.get('total_nf')}  Layout: {mt.get('layout')}")
    imp = nota.get('impostos',{})
    line(f"ICMS: {imp.get('icms')}  IPI: {imp.get('ipi')}  PIS: {imp.get('pis')}  COFINS: {imp.get('cofins')}")
    line("Itens (descrição / NCM / CFOP / vTotal):")
    for it in nota.get('itens', [])[:40]:
        line(f" - {it.get('descricao')} | NCM {it.get('ncm')} | CFOP {it.get('cfop')} | vTot {it.get('valor_total')}")
    c.save()

def export_pdf_auditoria(nota, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    c = canvas.Canvas(path, pagesize=A4); w, h = A4; y = h - 2*cm
    c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, "Relatório de Auditoria Fiscal"); y -= 1*cm
    c.setFont("Helvetica", 10)
    aud = audit_nota(nota)
    def line(txt):
        nonlocal y; c.drawString(2*cm, y, (txt or "")[:110]); y -= 0.6*cm
        if y < 2*cm: c.showPage(); y = h - 2*cm
    line(f"Emitente: {nota.get('emitente',{}).get('nome')}  CNPJ: {nota.get('emitente',{}).get('cnpj')}")
    line(f"Itens: {aud['resumo']['qtd_itens']}  Soma itens: {aud['resumo']['soma_itens']}  Total NF: {aud['resumo']['total_nf']}")
    line(f"CFOPs: {', '.join(aud['resumo']['cfops'])}")
    line(f"NCMs: {', '.join(aud['resumo']['ncms'])}")
    line("Inconsistências:")
    incs = aud["inconsistencias"] or ["Nenhuma inconsistência detectada."]
    for inc in incs: line(f"- {inc}")
    c.save()

def _run_streamlit_app():
    import streamlit as st
    from datetime import datetime
    import matplotlib.pyplot as plt
    from collections import Counter
    import csv, json, io

    st.set_page_config(page_title="AutonoMeta Extrator — Unificado", layout="wide", initial_sidebar_state="expanded")
    st.sidebar.title("AutonoMeta Extrator — Unificado")
    st.sidebar.caption("OCR + NLP + XML + Auditoria")
    st.sidebar.divider()
    st.sidebar.text_input("GOOGLE_API_KEY (opcional)", type="password", key="GOOGLE_API_KEY")
    st.sidebar.info("Sem chave, o parser usa heurística.")
    st.title("🧾 Extração de Dados Fiscais")
    st.write("Envie XML, PDF (multi-página) ou imagem.")

    uploaded = st.file_uploader("Arraste / clique (vários arquivos)", type=["xml","pdf","png","jpg","jpeg","tiff"], accept_multiple_files=True)

    def kpi_row(nota):
        total_nf = nota.get("metadados", {}).get("total_nf")
        imp = nota.get("impostos", {})
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Valor Total (NF)", f"{total_nf if total_nf is not None else '-'}")
        c2.metric("ICMS", f"{imp.get('icms','-')}")
        c3.metric("IPI", f"{imp.get('ipi','-')}")
        c4.metric("Itens", f"{len(nota.get('itens', []))}")

    def charts(nota):
        ncms = [i.get("ncm") for i in nota.get("itens", []) if i.get("ncm")]
        if ncms:
            cnt = Counter(ncms); labels, vals = zip(*cnt.items())
            fig = plt.figure(); plt.bar(labels, vals); plt.title("Itens por NCM"); st.pyplot(fig)
        cfops = [i.get("cfop") for i in nota.get("itens", []) if i.get("cfop")]
        if cfops:
            cnt = Counter(cfops); labels, vals = zip(*cnt.items())
            fig = plt.figure(); plt.bar(labels, vals); plt.title("Itens por CFOP"); st.pyplot(fig)

    results = []
    if uploaded:
        with st.spinner("Processando lote..."):
            for up in uploaded:
                tmp = f"/tmp/auto_{datetime.utcnow().timestamp():.0f}_{up.name}"
                with open(tmp, "wb") as f: f.write(up.read())
                res = process_document(tmp)
                results.append({"arquivo": up.name, **res})
        st.success(f"Processamento concluído ({len(results)} docs).")

        agg = aggregate_results(results)
        st.subheader("📊 Consolidado do Lote")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Documentos", agg["total_documentos"])
        c2.metric("Valor Total", f'{agg["valor_total"]:.2f}')
        c3.metric("Itens", agg["total_itens"])
        c4.metric("Docs com Problemas", agg["documentos_com_problemas"])
        c5.metric("ICMS Total", f'{agg["impostos_total"]["icms"]:.2f}')

        st.markdown("---")
        st.subheader("📄 Detalhes por Documento")
        for r in results:
            with st.expander(f'{r["arquivo"]} — {r["nota_fiscal"].get("emitente",{}).get("nome","(emitente desconhecido)")}'):
                st.json(r["nota_fiscal"])
                kpi_row(r["nota_fiscal"])
                charts(r["nota_fiscal"])

                colx, coly, colz, colw = st.columns(4)
                if colx.button(f"💾 Exportar JSON — {r['arquivo']}", key=f"json_{r['arquivo']}"):
                    st.download_button("Baixar JSON", data=json.dumps(r["nota_fiscal"], ensure_ascii=False, indent=2), file_name=f"{r['arquivo']}.json", mime="application/json")
                if coly.button(f"📑 Exportar CSV (itens) — {r['arquivo']}", key=f"csv_{r['arquivo']}"):
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=["descricao","ncm","cfop","quantidade","valor_unitario","valor_total"])
                    writer.writeheader()
                    for it in r["nota_fiscal"].get("itens", []):
                        writer.writerow({k: it.get(k) for k in ["descricao","ncm","cfop","quantidade","valor_unitario","valor_total"]})
                    st.download_button("Baixar CSV", data=buf.getvalue().encode("utf-8"), file_name=f"{r['arquivo']}_itens.csv", mime="text/csv")
                if colz.button(f"📄 Resumo Fiscal (PDF) — {r['arquivo']}", key=f"resumo_{r['arquivo']}"):
                    pdf_path = f"/tmp/resumo_{r['arquivo']}.pdf"
                    export_pdf_resumo(r["nota_fiscal"], pdf_path)
                    with open(pdf_path, "rb") as fh:
                        st.download_button("Baixar Resumo (PDF)", data=fh.read(), file_name=f"Resumo_{r['arquivo']}.pdf", mime="application/pdf")
                if colw.button(f"🧪 Auditoria (PDF) — {r['arquivo']}", key=f"audit_{r['arquivo']}"):
                    pdf_path = f"/tmp/auditoria_{r['arquivo']}.pdf"
                    export_pdf_auditoria(r["nota_fiscal"], pdf_path)
                    with open(pdf_path, "rb") as fh:
                        st.download_button("Baixar Auditoria (PDF)", data=fh.read(), file_name=f"Auditoria_{r['arquivo']}.pdf", mime="application/pdf")

def _run_cli(argv):
    if len(argv) < 2:
        print("Uso: python extrator_unificado.py <arquivo.xml|pdf|imagem>"); sys.exit(1)
    path = argv[1]; out = process_document(path)
    print(json.dumps(out["nota_fiscal"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if os.environ.get("STREAMLIT_EXECUTION_CONTEXT"):
        pass
    elif "streamlit" in sys.argv[0].lower():
        pass
    elif len(sys.argv) > 1:
        _run_cli(sys.argv)
    else:
        try:
            import streamlit as st  # noqa
            _run_streamlit_app()
        except Exception:
            print("Instale streamlit para a UI ou forneça um arquivo como argumento para modo CLI.")
