import logging
import os
import sys
import uuid
from typing import List, Tuple

# Blocage strict des connexions sortantes
os.environ["HF_HUB_OFFLINE"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# DÉTERMINATION DU CHEMIN DE BASE (PORTABILITÉ .EXE)
# ──────────────────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb

# Validation de la dépendance PDF
try:
    import pypdf
except ImportError:
    raise ImportError("Le module pypdf est manquant. Exécuter : pip install pypdf")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION ET LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("SKY_MEMORY")

# Application du chemin relatif dynamique pour ChromaDB
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_memory")
COLLECTION_NAME = "long_term_memory"

# Taxonomie stricte des domaines de connaissances
DOMAINS = [
    "Mémoire scientifique",
    "Mémoire historique",
    "Mémoire informatique",
    "Mémoire philosophie",
    "Mémoire personnelle"
]

encoder_instance = None
chroma_collection = None

# ──────────────────────────────────────────────────────────────────────────────
# INITIALISATION DES RESSOURCES LOCALES
# ──────────────────────────────────────────────────────────────────────────────
def get_resources() -> Tuple[SentenceTransformer, chromadb.Collection]:
    """Charge le modèle d'embedding hors ligne et connecte la base de données."""
    global encoder_instance, chroma_collection
    if encoder_instance is None or chroma_collection is None:
        # Détermination du chemin absolu local vers le modèle d'embedding
        embedding_path = os.path.join(BASE_DIR, "embedding_model")
        logger.info(f"Chargement du modèle d'embedding local depuis : {embedding_path}")
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Modèle d'embedding introuvable à l'emplacement : {embedding_path}")
            
        # Chargement forcé via le répertoire local sans dépendance au cache utilisateur
        encoder_instance = SentenceTransformer(embedding_path, local_files_only=True)
        
        logger.info(f"Connexion à l'instance ChromaDB : {CHROMA_DB_DIR}")
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
        
    return encoder_instance, chroma_collection

# ──────────────────────────────────────────────────────────────────────────────
# TRAITEMENT ET SEGMENTATION DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Segmente le texte en blocs pour préserver la densité vectorielle."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# ──────────────────────────────────────────────────────────────────────────────
# OPÉRATIONS D'INGESTION VECTORIELLE
# ──────────────────────────────────────────────────────────────────────────────
def inject_knowledge(domain: str, raw_text: str, file_obj) -> str:
    """Vectorise et indexe les données sémantiques avec métadonnées structurelles."""
    content = ""
    
    # 1. Extraction des données
    if file_obj is not None:
        try:
            file_extension = os.path.splitext(file_obj.name)[1].lower()
            if file_extension == '.pdf':
                reader = pypdf.PdfReader(file_obj.name)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        content += extracted_text + "\n"
            else:
                with open(file_obj.name, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as e:
            return f"Erreur de lecture du fichier : {e}"
    elif raw_text.strip():
        content = raw_text.strip()
    else:
        return "Erreur : Aucune donnée textuelle ou fichier fourni."

    # 2. Segmentation spatiale
    chunks = chunk_text(content)
    if not chunks:
        return "Erreur : Contenu inexploitable après segmentation."

    # 3. Vectorisation et indexation
    try:
        encoder, collection = get_resources()
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"kb_{uuid.uuid4().hex[:12]}_p{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(encoder.encode(chunk).tolist())
            
            # Injection des métadonnées strictes
            metadatas.append({
                "source": "knowledge_injector",
                "type": "knowledge_base",
                "domain": domain
            })

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Ingestion réussie : {len(chunks)} vecteurs ajoutés au domaine '{domain}'.")
        return f"[SUCCÈS] {len(chunks)} segments vectoriels insérés dans la catégorie : {domain}."
        
    except Exception as e:
        logger.error(f"Échec de l'indexation vectorielle : {e}")
        return f"[ÉCHEC CRITIQUE] Erreur lors de l'insertion dans ChromaDB : {e}"

# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE D'ADMINISTRATION DES CONNAISSANCES
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="SKY - Gestionnaire de Mémoire Sémantique") as demo:
    gr.Markdown("# Interface d'Ingestion de la Base de Connaissances")
    gr.Markdown(
        "Ce module insère des données factuelles directement dans l'espace vectoriel "
        "`long_term_memory` de ChromaDB. Les formats `.txt` et `.pdf` sont pris en charge."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            domain_radio = gr.Radio(choices=DOMAINS, label="Domaine d'allocation", value=DOMAINS[0])
            file_input = gr.File(label="Importation de document (.txt, .pdf)", file_types=[".txt", ".pdf"])
            text_input = gr.Textbox(label="Insertion de texte brut", lines=10, placeholder="Saisir la documentation ou les faits à archiver...")
            submit_btn = gr.Button("Indexer dans la Mémoire", variant="primary")
            
        with gr.Column(scale=1):
            out_status = gr.Textbox(label="Statut de la transaction vectorielle", lines=4, interactive=False)

    submit_btn.click(
        fn=inject_knowledge,
        inputs=[domain_radio, text_input, file_input],
        outputs=[out_status]
    )

if __name__ == "__main__":
    logger.info("Pré-chargement des ressources vectorielles...")
    get_resources()
    logger.info("Démarrage du service d'ingestion sur le port 7862.")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False
    )

