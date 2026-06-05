import logging
import os
import re
import sys
import uuid
from typing import Dict, List, Tuple, Generator

# Désactivation absolue des appels réseau sortants
os.environ["HF_HUB_OFFLINE"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# DÉTERMINATION DU CHEMIN DE BASE (PORTABILITÉ .EXE)
# ──────────────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import gradio as gr
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb

# Validation des dépendances scientifiques obligatoires
try:
    import numpy as np
    import sympy as sp
except ImportError:
    raise ImportError("Composants scientifiques manquants. Exécuter : pip install numpy sympy")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION SYSTEME ET LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("SKY_EPISTEMIC")

# Application des chemins relatifs dynamiques
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "Llama-3.1-8B-Abliterated.Q4_K_M.gguf")
MODEL_PATH = os.environ.get("SKY_MODEL_PATH", DEFAULT_MODEL)
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_memory")

N_CTX = 8192  
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

FN_EXTRACTEUR = "Étape 1 : Extraction Atomique des Affirmations"
FN_AXIOMES = "Étape 2 : Détection des Axiomes Implicites (Contextualisée)"
FN_SANDBOX = "Étape 3 : Validation Empirique en Sandbox Spécifique"
FN_EVALUATEUR = "Étape 4 : Matrice de Confiance Épistémique Finale"

DOMAINS = [
    "Mémoire scientifique",
    "Mémoire historique",
    "Mémoire informatique",
    "Mémoire philosophie",
    "Mémoire personnelle"
]

encoder_instance = None

# ──────────────────────────────────────────────────────────────────────────────
# INITIALISATION DES DRIVERS LOCAUX
# ──────────────────────────────────────────────────────────────────────────────
def get_llm() -> Llama:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Composant d'inférence introuvable : {MODEL_PATH}")
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        chat_format="llama-3",
        verbose=False
    )

def get_chroma_collection():
    """Charge le modèle d'embedding hors ligne local et connecte la base vectorielle."""
    global encoder_instance
    if encoder_instance is None:
        # Détermination du chemin absolu local vers le modèle d'embedding (aligné sur main.py et memory.py)
        embedding_path = os.path.join(BASE_DIR, "embedding_model")
        logger.info(f"Chargement du modèle d'embedding local depuis : {embedding_path}")
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Modèle d'embedding introuvable à l'emplacement : {embedding_path}")
            
        # Chargement forcé via le répertoire local sans dépendance au cache utilisateur
        encoder_instance = SentenceTransformer(embedding_path, local_files_only=True)
        
    logger.info(f"Connexion à l'instance ChromaDB : {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="long_term_memory")
    return encoder_instance, collection

def query_domain_knowledge(query_text: str, domain: str, n_results: int = 3) -> str:
    """Récupère les connaissances certifiées du domaine ciblé pour orienter l'audit."""
    try:
        encoder, collection = get_chroma_collection()
        if collection.count() == 0:
            return ""
            
        query_embedding = encoder.encode(query_text).tolist()
        
        where_filter = {
            "$and": [
                {"type": "knowledge_base"},
                {"domain": domain}
            ]
        }
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""
            
        return "\n\n---\n\n".join(documents)
    except Exception as e:
        logger.error(f"Échec de la récupération vectorielle : {e}")
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# MOTEUR DE LA SANDBOX NATIVE RESTREINTE
# ──────────────────────────────────────────────────────────────────────────────
def safe_execute_code(code_str: str) -> str:
    """Isole et exécute le code de vérification scientifique en environnement fermé."""
    if not code_str.strip():
        return "Aucun code généré pour la validation."

    if "```python" in code_str:
        code_clean = code_str.split("```python")[1].split("```")[0].strip()
    else:
        code_clean = re.sub(r"```", "", code_str).strip()

    from io import StringIO
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output

    safe_globals = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "chr": chr,
            "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "format": format, "hash": hash, "hex": hex, "id": id,
            "int": int, "isinstance": isinstance, "issubclass": issubclass, "len": len,
            "list": list, "map": map, "max": max, "min": min, "next": next, "object": object,
            "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set, "slice": slice,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip
        },
        "np": np,
        "sp": sp
    }
    
    local_vars = {}
    error_output = ""

    try:
        exec(code_clean, safe_globals, local_vars)
    except Exception as e:
        error_output = str(e)
    finally:
        sys.stdout = old_stdout

    stdout_result = redirected_output.getvalue().strip()
    
    if error_output:
        return f"[ÉCHEC LOGICIEL]\nErreur renvoyée : {error_output}\nCode exécuté :\n{code_clean}"
    
    if not stdout_result and local_vars:
        stdout_result = "\n".join([f"{k} = {v}" for k, v in local_vars.items() if not k.startswith("__")])
        
    return f"[SUCCÈS OPÉRATIONNEL]\n\nSortie d'exécution :\n{stdout_result if stdout_result else 'Calcul effectué sans sortie texte.'}"

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS SYSTEMES DE RIGUEUR COGNITIVE
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_EXTRACTEUR = (
    "Tu es un analyseur syntaxique et logique froid. Ta tâche est d'isoler uniquement "
    "les affirmations déclaratives, thèses, équations ou postulats présents dans le texte.\n"
    "CONTRAINTES STRICTES :\n"
    "- Exclus l'enrobage narratif.\n"
    "- Produis uniquement une liste numérotée de propositions atomiques autonomes."
)

PROMPT_AXIOMES = (
    "Tu es un détecteur de présuppositions logiques. Tu reçois une liste d'affirmations ainsi qu'un "
    "corpus de connaissances certifiées du domaine.\n"
    "Ta tâche est d'identify les axiomes implicites ou conditions aux limites "
    "non formulés par l'auteur, en t'appuyant sur les connaissances de référence si elles sont pertinentes.\n"
    "CONTRAINTES STRICTES :\n"
    "- Restes purement analytique et technique."
)

PROMPT_GEN_CODE = (
    "Tu es un ingénieur de validation empirique. Tu reçois des thèses, des axiomes et des références certifiées.\n"
    "Ta tâche exclusive est de rédiger un code Python strict pour modéliser et vérifier numériquement "
    "ou formellement la véracité de ces affirmations. Inspire-toi des références pour la méthodologie.\n"
    "CONTRAINTES STRICTES :\n"
    "- Utilise 'np' pour NumPy ou 'sp' pour SymPy.\n"
    "- Affiche le résultat final ou la preuve avec 'print()'.\n"
    "- Génère uniquement le bloc de code Python dans des balises markdown."
)

PROMPT_EVALUATEUR = (
    "Tu es un estimateur métrique de fiabilité épistémique. Tu reçois les affirmations, "
    "les axiomes cachés et les résultats d'exécution réels de la sandbox.\n"
    "Ta tâche est de dresser la matrice de confiance synthétique finale.\n"
    "CONTRAINTES DE FORMATAGE STRICT : Réponds uniquement selon cette structure :\n"
    "## 1. INVENTAIRE DES POSTULATS FRAGILES ET DES ETATS DE CODE\n"
    "[Lister les faiblesses logiques]\n\n"
    "## 2. SCORE DE VALIDITÉ LOGIQUE\n"
    "Note : [X/10]\n\n"
    "## 3. THÉORIE RÉVISÉE ET CONCLUSION\n"
    "[Rédiger la contre-théorie objective finale]"
)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITAIRES DE RENDU VISUEL (HTML/CSS)
# ──────────────────────────────────────────────────────────────────────────────
def generate_visual_indicator(eval_text: str) -> str:
    match = re.search(r"Note\s*:\s*\[?(\d+)/10\]?", eval_text)
    if not match:
        return "<div style='padding:10px; background:#222; border-radius:4px;'>Indicateur de validation non généré.</div>"
    
    score = int(match.group(1))
    percentage = score * 10
    
    if score >= 7:
        color = "#2ecc71"
        status = "CRITIQUE : APTE (VALIDATION EXPERIMENTALE EN SANDBOX REUSSIE)"
    elif score >= 5:
        color = "#f1c40f"
        status = "CRITIQUE : VIGILANCE (CALCULS INSTABLES OU INCOMPLETS)"
    else:
        color = "#e74c3c"
        status = "CRITIQUE : REJETÉ (ANOMALIE DE CALCUL OU DE LOGIQUE INTERNE)"
        
    html = f"""
    <div style="background-color: #1a1a1a; padding: 20px; border-radius: 8px; border-left: 5px solid {color}; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: bold; color: #ffffff; font-family: monospace; font-size: 14px;">METRIQUE D'INTEGRITE EPISTEMIQUE & SANDBOX</span>
            <span style="background-color: {color}; color: #000000; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; font-family: monospace;">{status}</span>
        </div>
        <div style="font-size: 36px; font-weight: 900; color: {color}; font-family: sans-serif; margin-bottom: 10px;">
            {score} <span style="font-size: 18px; color: #888888;">/ 10</span>
        </div>
        <div style="background-color: #333333; border-radius: 4px; height: 12px; width: 100%; overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {percentage}%; transition: width 0.5s ease-in-out;"></div>
        </div>
    </div>
    """
    return html

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE COGNITIF INTEGRAL (AVEC RAG)
# ──────────────────────────────────────────────────────────────────────────────
def execute_cognitive_step(model: Llama, system_prompt: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    output = model.create_chat_completion(
        messages=messages,
        max_tokens=1536,
        temperature=0.1,
        stream=False
    )
    return output["choices"][0]["message"]["content"].strip()


def run_cognitive_pipeline(raw_text: str, domain: str) -> Generator[Tuple[str, str, str, str, str], None, None]:
    if not raw_text.strip():
        yield ("Erreur : Flux vide.", "", "", "", "")
        return

    logger.info("Sollicitation du moteur GGUF et du réseau vectoriel...")
    
    # 0. Récupération des connaissances de référence (RAG interne)
    domain_knowledge = query_domain_knowledge(raw_text, domain)
    if domain_knowledge:
        logger.info(f"Connaissances contextuelles ({domain}) injectées avec succès.")
    
    model = get_llm()

    extract_res = "Isolement syntaxique des formules et assertions en cours..."
    axiom_res = "En attente du signal d'extraction..."
    sandbox_res = "En attente de la construction algorithmique..."
    eval_res = "En attente de la clôture des calculs..."
    visual_html = "<div style='color: #666;'>Simulation en cours...</div>"
    
    yield (extract_res, axiom_res, sandbox_res, eval_res, visual_html)

    # Étape 1 : Extraction
    logger.info(f"Exécution : {FN_EXTRACTEUR}")
    extract_res = execute_cognitive_step(model, PROMPT_EXTRACTEUR, f"Données brutes :\n{raw_text}")
    axiom_res = "Confrontation avec la base de connaissances certifiée..."
    yield (extract_res, axiom_res, sandbox_res, eval_res, visual_html)
    
    # Étape 2 : Axiomes (Contextualisée par la mémoire)
    logger.info(f"Exécution : {FN_AXIOMES}")
    input_axiomes = f"ÉLÉMENTS EXTRAITS :\n{extract_res}\n\n[CONNAISSANCES CERTIFIÉES DE RÉFÉRENCE] :\n{domain_knowledge if domain_knowledge else 'Aucune référence spécifique.'}"
    axiom_res = execute_cognitive_step(model, PROMPT_AXIOMES, input_axiomes)
    sandbox_res = "Synthèse et écriture du script Python d'évaluation empirique..."
    yield (extract_res, axiom_res, sandbox_res, eval_res, visual_html)
    
    # Étape 3 : Génération de code et exécution Sandbox
    logger.info("Compilation du script de validation...")
    input_code_gen = f"ASSERTIONS CONSTITUTIVES :\n{extract_res}\n\nAXIOMES ET LIMITES :\n{axiom_res}\n\n[RÉFÉRENCES MÉTHODOLOGIQUES] :\n{domain_knowledge if domain_knowledge else 'N/A'}"
    generated_code = execute_cognitive_step(model, PROMPT_GEN_CODE, input_code_gen)
    
    logger.info(f"Exécution : {FN_SANDBOX}")
    execution_result = safe_execute_code(generated_code)
    
    sandbox_res = f"### SCRIPT FORMEL GENERATION\n```python\n{generated_code.replace('```python', '').replace('```', '').strip()}\n```\n\n### RAPPORT D'EXECUTION FLUX SORTANT\n{execution_result}"
    eval_res = "Construction de l'évaluation finale..."
    yield (extract_res, axiom_res, sandbox_res, eval_res, visual_html)
    
    # Étape 4 : Évaluation métrique
    logger.info(f"Exécution : {FN_EVALUATEUR}")
    input_eval = (
        f"THESES EXTRAITES :\n{extract_res}\n\n"
        f"AXIOMES ASSOCIES :\n{axiom_res}\n\n"
        f"COMPTE RENDU EXECUTIF DE LA SANDBOX :\n{execution_result}"
    )
    eval_res = execute_cognitive_step(model, PROMPT_EVALUATEUR, input_eval)
    
    visual_html = generate_visual_indicator(eval_res)
    yield (extract_res, axiom_res, sandbox_res, eval_res, visual_html)

    # Étape 5 : Persistance vectorielle (Indexation comme connaissance certifiée)
    logger.info(f"Transmission à l'espace de stockage ChromaDB (Domaine: {domain})...")
    try:
        encoder, collection = get_chroma_collection()
        doc_id = f"epistemic_matrix_{str(uuid.uuid4())[:8]}"
        embedding = encoder.encode(eval_res).tolist()
        
        # Le rapport de validation est maintenant injecté en tant que "knowledge_base" pour être utilisé par main.py
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[eval_res],
            metadatas=[{
                "source": "introspection_module",
                "type": "knowledge_base",
                "domain": domain,
                "integrity_hash": str(hash(extract_res[:50]))
            }]
        )
        logger.info(f"Indexation finalisée. Identifiant certifié : {doc_id}")
    except Exception as e:
        logger.error(f"Erreur d'écriture base vectorielle : {e}")


def process_input(file_obj, text_area, domain):
    if file_obj is not None:
        try:
            with open(file_obj.name, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            yield (f"Erreur d'accès matériel : {e}", "", "", "", "")
            return
    else:
        content = text_area

    if not content.strip():
        yield ("Erreur : Flux d'entrée nul.", "", "", "", "")
        return
        
    for current_state in run_cognitive_pipeline(content, domain):
        yield current_state

# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE GRAPHIQUE
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Analyseur Épistémique & Sandbox") as demo:
    gr.Markdown("# Déconstruction Épistémique Autonome & Validation en Sandbox")
    gr.Markdown(
        "Ce module isole les thèses, s'appuie sur la base de connaissances certifiée, génère une simulation mathématique "
        "puis exécute le code en environnement natif restreint. La matrice d'évaluation finale est enregistrée "
        "en tant que nouvelle connaissance absolue."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            domain_dropdown = gr.Dropdown(
                choices=DOMAINS,
                value=DOMAINS[0],
                label="Domaine d'analyse (Ciblage RAG & Archivage)"
            )
            file_input = gr.File(label="Fichier Source (.txt)", file_types=[".txt"])
            text_input = gr.Textbox(label="Injection de texte ou équations", lines=10, placeholder="Saisir la théorie ou le bloc de formules...")
            submit_btn = gr.Button("Lancer le Diagnostic Empirique", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Accordion(FN_EXTRACTEUR, open=False):
                out_extract = gr.Markdown("En attente de traitement...")
            with gr.Accordion(FN_AXIOMES, open=False):
                out_hypo = gr.Markdown("En attente de traitement...")
            with gr.Accordion(FN_SANDBOX, open=True):
                out_sandbox = gr.Markdown("En attente de l'exécution des scripts...")
            with gr.Accordion(FN_EVALUATEUR, open=True):
                out_visual = gr.HTML("<div style='color: #666;'>En attente des données critiques...</div>")
                out_evaluation = gr.Markdown("En attente du verdict...")

    submit_btn.click(
        fn=process_input,
        inputs=[file_input, text_input, domain_dropdown],
        outputs=[out_extract, out_hypo, out_sandbox, out_evaluation, out_visual]
    )

if __name__ == "__main__":
    # Pré-chargement des ressources vectorielles au lancement du script
    logger.info("Pré-chargement des ressources vectorielles...")
    get_chroma_collection()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )

