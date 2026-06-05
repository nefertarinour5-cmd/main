import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, List

# Blocage strict des requêtes réseau externes
os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb

# ──────────────────────────────────────────────────────────────────────────────
# DÉTERMINATION DU CHEMIN DE BASE (PORTABILITÉ .EXE)
# ──────────────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION ET LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("SKY_MAIN")

# Application des chemins relatifs dynamiques
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "Llama-3.1-8B-Abliterated.Q4_K_M.gguf")
MODEL_PATH = os.environ.get("SKY_MODEL_PATH", DEFAULT_MODEL)
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_memory")

N_CTX = 8192
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
MAX_TOKENS = 1024
TEMPERATURE = 0.7

llm_instance = None
encoder_instance = None
chroma_collection = None

load_lock = threading.Lock()
generation_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# INITIALISATION DES DRIVERS
# ──────────────────────────────────────────────────────────────────────────────
def get_llm() -> Llama:
    global llm_instance
    if llm_instance is None:
        with load_lock:
            if llm_instance is None:
                logger.info(f"Allocation du modèle GGUF : {MODEL_PATH}")
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")
                llm_instance = Llama(
                    model_path=MODEL_PATH,
                    n_ctx=N_CTX,
                    n_threads=N_THREADS,
                    chat_format="llama-3",
                    verbose=False
                )
    return llm_instance

def get_rag_system():
    """Charge le modèle d'embedding hors ligne local et connecte la base vectorielle."""
    global encoder_instance, chroma_collection
    if encoder_instance is None or chroma_collection is None:
        with load_lock:
            if encoder_instance is None:
                # Détermination du chemin absolu local vers le modèle d'embedding
                embedding_path = os.path.join(BASE_DIR, "embedding_model")
                logger.info(f"Chargement du modèle d'embedding local depuis : {embedding_path}")
                
                if not os.path.exists(embedding_path):
                    raise FileNotFoundError(f"Modèle d'embedding introuvable à l'emplacement : {embedding_path}")
                
                # Chargement forcé via le répertoire local sans dépendance au cache utilisateur
                encoder_instance = SentenceTransformer(embedding_path, local_files_only=True)
                
            if chroma_collection is None:
                logger.info(f"Connexion à l'instance ChromaDB persistante : {CHROMA_DB_DIR}")
                client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                chroma_collection = client.get_or_create_collection(name="long_term_memory")
    return encoder_instance, chroma_collection

# ──────────────────────────────────────────────────────────────────────────────
# OPÉRATIONS VECTORIELLES AVEC FILTRAGE AVANCÉ
# ──────────────────────────────────────────────────────────────────────────────
def query_memory(query_text: str, search_mode: str, n_results: int = 3) -> str:
    """Recherche par similarité cosinus avec application de filtres sur les métadonnées."""
    try:
        encoder, collection = get_rag_system()
        if collection.count() == 0:
            return ""
            
        query_embedding = encoder.encode(query_text).tolist()
        
        # Application du routage contextuel
        where_filter = None
        if search_mode != "Mémoire globale (Tout)":
            where_filter = {"domain": search_mode}
            
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

def save_chat_to_memory(user_msg: str, assistant_msg: str):
    """Vectorise et archive la transaction courante en tant que Mémoire personnelle."""
    try:
        encoder, collection = get_rag_system()
        if not user_msg or not assistant_msg:
            return

        text_to_embed = f"Utilisateur : {user_msg}\nAssistant : {assistant_msg}"
        embedding = encoder.encode(text_to_embed).tolist()
        doc_id = f"chat_personal_{str(uuid.uuid4())[:12]}"
        
        # Assignation stricte au domaine personnel
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text_to_embed],
            metadatas=[{
                "source": "chat_interface", 
                "type": "chat_history",
                "domain": "Mémoire personnelle"
            }]
        )
        logger.info(f"Transaction archivée (Mémoire personnelle). ID: {doc_id}")
    except Exception as e:
        logger.error(f"Échec de l'écriture vectorielle : {e}")

# ──────────────────────────────────────────────────────────────────────────────
# FORMATAGE ET INFÉRENCE
# ──────────────────────────────────────────────────────────────────────────────
def normalize_history(history: Any) -> List[Dict[str, str]]:
    """Standardise l'historique Gradio au format OpenAI."""
    messages: List[Dict[str, str]] = []
    if not history:
        return messages

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg not in (None, ""):
                messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg not in (None, ""):
                messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages

def stream_inference(message: str, history: Any, search_mode: str):
    """Exécute l'inférence avec injection du contexte vectoriel filtré."""
    model = get_llm()
    
    # Extraction du contexte pertinent selon le domaine sélectionné
    retrieved_context = query_memory(message, search_mode, n_results=3)
    
    # Construction du prompt système
    system_content = (
        "Vous êtes une intelligence artificielle d'analyse fonctionnant sur l'architecture Sky. "
        "Vous êtes précis, factuel et direct.\n"
    )
    
    if retrieved_context:
        system_content += (
            f"\n[DÉBUT DES ARCHIVES : {search_mode.upper()}]\n"
            f"{retrieved_context}\n"
            "[FIN DES ARCHIVES]\n"
            "Utilisez ces données contextuelles si elles sont pertinentes pour traiter la requête courante."
        )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        with generation_lock:
            stream = model.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    response += token
                    yield response
                    
        # Exécution de l'archivage asynchrone post-génération
        save_chat_to_memory(message, response)

    except Exception as e:
        logger.exception("Exception levée durant l'inférence.")
        yield f"\n[Erreur de traitement : {e}]"

def predict_interface(message: str, history: Any, search_mode: str):
    logger.info(f"Requête entrante détectée. Filtre actif : {search_mode}")
    for partial in stream_inference(message, history, search_mode):
        yield partial

# ──────────────────────────────────────────────────────────────────────────────
# EXÉCUTION DU SERVEUR LOCAL ET INTERFACE
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Interface Principale SKY") as demo:
    gr.Markdown("# Terminal de Commande Principal")
    gr.Markdown("Interface couplée à la mémoire persistante ChromaDB avec routage par domaine.")
    
    # Définition du paramètre de filtrage
    search_mode_dropdown = gr.Dropdown(
        choices=[
            "Mémoire globale (Tout)",
            "Mémoire scientifique",
            "Mémoire historique",
            "Mémoire informatique",
            "Mémoire philosophie",
            "Mémoire personnelle"
        ],
        value="Mémoire globale (Tout)",
        label="Ciblage du contexte vectoriel"
    )
    
    # Injection du paramètre supplémentaire dans l'interface de chat
    gr.ChatInterface(
        fn=predict_interface,
        additional_inputs=[search_mode_dropdown],
        autofocus=True
    )

if __name__ == "__main__":
    logger.info("Initialisation préalable du sous-système vectoriel.")
    get_rag_system()
    
    logger.info("Démarrage du service d'interface sur le port 7860.")
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
