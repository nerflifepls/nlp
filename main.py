import numpy as np
import pandas as pd
import torch 
import gensim
import gensim.downloader as api 
import transformers
import sentence_transformers
import nltk
import spacy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

import language_tool_python

from spellchecker import SpellChecker 
from nltk.tokenize.treebank import TreebankWordDetokenizer


from transformers import pipeline


import sys
logger = logging.getLogger()

logger.setLevel(logging.INFO)

# Check if handlers already exist (e.g., from imported libraries or running in interactive mode)
if not logger.hasHandlers():

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s') # Added filename/line
    console_handler = logging.StreamHandler(sys.stdout) 
    console_handler.setLevel(logging.INFO) 
    console_handler.setFormatter(log_formatter)

    logger.addHandler(console_handler)
    logging.info("--- Explicit logger configured successfully ---") 
else:
    logging.info("--- Logger already has handlers. Using existing configuration. ---")


SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SPACY_MODEL_LG = "en_core_web_lg"
SPACY_MODEL_SM = "en_core_web_sm"
WORD2VEC_MODEL_NAME = "word2vec-google-news-300"
FASTTEXT_MODEL_NAME = "fasttext-wiki-news-subwords-300"
GREEK_BERT_MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
GREEK_SPACY_MODEL = "el_core_news_sm"

MASK_PLACEHOLDER_GR = "ΜΑΣΚΑ"

def download_nltk_resources():
    """Downloads necessary NLTK resources if not found."""
    resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords'] 
    for resource in resources:
        try:
            if resource == 'punkt':
                 nltk.data.find('tokenizers/punkt')
            elif resource == 'averaged_perceptron_tagger':
                 nltk.data.find('taggers/averaged_perceptron_tagger')
            elif resource == 'stopwords':
                 nltk.data.find('corpora/stopwords')
            logging.debug(f"NLTK resource '{resource}' found.")
        except nltk.downloader.DownloadError:
            logging.info(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)
            logging.info(f"'{resource}' downloaded.")
        except LookupError: # Sometimes find raises LookupError
             logging.info(f"NLTK resource '{resource}' not found (LookupError). Downloading...")
             nltk.download(resource, quiet=True)
             logging.info(f"'{resource}' downloaded.")


TEXT_1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

TEXT_2 = """During our final discuss, I told him about the new submission – the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn't see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""

nlp_spacy = None
ACTUAL_SPACY_DIM = 0
lang_tool = None
sbert_model = None
SBERT_DIM = 0
word2vec_model = None
fasttext_model = None
nlp_greek = None
fill_mask_pipeline_greek = None

# --- Model Loading Function ---
def load_models():
    """Loads all required NLP models globally."""
    global nlp_spacy, ACTUAL_SPACY_DIM, lang_tool, sbert_model, SBERT_DIM
    global word2vec_model, fasttext_model, nlp_greek, fill_mask_pipeline_greek

    logging.info("--- Starting Model Loading ---")

    # SpaCy (English)
    try:
        nlp_spacy = spacy.load(SPACY_MODEL_LG)
        ACTUAL_SPACY_DIM = nlp_spacy.vocab.vectors_length if nlp_spacy.vocab.vectors_length > 0 else 300
        logging.info(f"SpaCy '{SPACY_MODEL_LG}' loaded. Vector dim: {ACTUAL_SPACY_DIM}")
    except OSError:
        logging.warning(f"SpaCy '{SPACY_MODEL_LG}' not found. Trying '{SPACY_MODEL_SM}'.")
        try:
            nlp_spacy = spacy.load(SPACY_MODEL_SM)
            # Sm model might have vectors (96) or not (0). Default to 96 if none.
            ACTUAL_SPACY_DIM = nlp_spacy.vocab.vectors_length if nlp_spacy.vocab.vectors_length > 0 else 96
            logging.info(f"SpaCy '{SPACY_MODEL_SM}' loaded. Vector dim: {ACTUAL_SPACY_DIM}")
            if ACTUAL_SPACY_DIM <= 0:
                 logging.warning("Loaded SpaCy model has no word vectors. SpaCy embeddings will be zero vectors.")
        except OSError:
            logging.error(f"SpaCy '{SPACY_MODEL_SM}' also not found. SpaCy features limited.")
            ACTUAL_SPACY_DIM = 96 # Assume a fallback dimension if needed elsewhere

    # LanguageTool
    try:
        lang_tool = language_tool_python.LanguageTool('en-US')
        logging.info("LanguageTool ('en-US') loaded successfully.")
    except Exception as e:
        logging.error(f"Could not load LanguageTool: {e}. Reconstruction B1 might fail.")

    try:
        sbert_model = sentence_transformers.SentenceTransformer(SBERT_MODEL_NAME)
        SBERT_DIM = sbert_model.get_sentence_embedding_dimension()
        logging.info(f"SentenceTransformer '{SBERT_MODEL_NAME}' loaded. Vector dim: {SBERT_DIM}")
    except Exception as e:
        logging.error(f"Could not load SentenceTransformer model '{SBERT_MODEL_NAME}': {e}")
        SBERT_DIM = 384

    # Word2Vec
    try:
        logging.info(f"Loading Word2Vec model ({WORD2VEC_MODEL_NAME})... This might take a while.")
        word2vec_model = api.load(WORD2VEC_MODEL_NAME)
        logging.info(f"Word2Vec model loaded. Vector dim: {word2vec_model.vector_size}")
    except Exception as e:
        logging.error(f"Could not load Word2Vec model '{WORD2VEC_MODEL_NAME}': {e}")

    # FastText
    try:
        logging.info(f"Loading FastText model ({FASTTEXT_MODEL_NAME})... This might take a while.")
        fasttext_model = api.load(FASTTEXT_MODEL_NAME)
        logging.info(f"FastText model loaded. Vector dim: {fasttext_model.vector_size}")
    except Exception as e:
        logging.error(f"Could not load FastText model '{FASTTEXT_MODEL_NAME}': {e}")

    # SpaCy (Greek) - for Bonus
    try:
        nlp_greek = spacy.load(GREEK_SPACY_MODEL)
        logging.info(f"SpaCy Greek model '{GREEK_SPACY_MODEL}' loaded.")
    except OSError:
        logging.warning(f"SpaCy Greek model '{GREEK_SPACY_MODEL}' not found.")
        logging.warning("Please run: python -m spacy download el_core_news_sm")
        logging.warning("Greek syntactic analysis for Bonus will be skipped.")

    # Fill-Mask Pipeline (Greek) - for Bonus
    try:
        fill_mask_pipeline_greek = pipeline("fill-mask", model=GREEK_BERT_MODEL_NAME, top_k=5)
        logging.info(f"Greek fill-mask pipeline loaded with model: {GREEK_BERT_MODEL_NAME}")
    except Exception as e:
        logging.error(f"Could not load Greek fill-mask pipeline ('{GREEK_BERT_MODEL_NAME}'): {e}")

    logging.info("--- Model Loading Complete ---")


# --- Helper Functions ---
def get_sentence_embedding_sbert(text: str, model: sentence_transformers.SentenceTransformer) -> np.ndarray:
    """Encodes text using a SentenceTransformer model. Returns zero vector on failure."""
    global SBERT_DIM
    if model is None:
        logging.warning("SBERT model unavailable. Returning zero vector.")
        return np.zeros(SBERT_DIM if SBERT_DIM > 0 else 384)
    try:
        return model.encode(text)
    except Exception as e:
        logging.error(f"Error encoding text with SBERT: {e}")
        return np.zeros(SBERT_DIM if SBERT_DIM > 0 else 384)

def get_sentence_embedding_spacy(text: str, nlp_model: spacy.language.Language) -> np.ndarray:
    """Generates sentence embedding using SpaCy's doc.vector. Returns zero vector on failure or if no vectors."""
    global ACTUAL_SPACY_DIM
    fallback_dim = ACTUAL_SPACY_DIM if ACTUAL_SPACY_DIM > 0 else 96
    if nlp_model is None or not nlp_model.vocab.vectors_length > 0:
        return np.zeros(fallback_dim)
    try:
        doc = nlp_model(text)
        return doc.vector if doc.has_vector else np.zeros(fallback_dim)
    except Exception as e:
        logging.error(f"Error getting SpaCy vector: {e}")
        return np.zeros(fallback_dim)


def get_sentence_embedding_gensim_avg(text: str, gensim_model) -> np.ndarray:
    """
    Generates sentence embedding by averaging word vectors from a Gensim model.
    Returns zero vector if model unavailable or no words found in vocab.
    """
    if gensim_model is None:
        logging.warning("Gensim model unavailable. Returning zero vector.")
        return np.array([])

    vec_size = gensim_model.vector_size
    try:
        words = nltk.word_tokenize(text.lower())
        word_vectors = [gensim_model[word] for word in words if word in gensim_model.key_to_index]

        if not word_vectors:
            return np.zeros(vec_size)
        return np.mean(word_vectors, axis=0)
    except Exception as e:
        logging.error(f"Error getting Gensim average vector: {e}")
        return np.zeros(vec_size)


def calculate_cosine_similarity_scores(original_text: str, reconstructed_text: str,
                                       embedding_fn: callable, model_instance) -> float:
    """
    Calculates cosine similarity between original and reconstructed text.
    Uses average sentence similarity if sentence counts match, else full text similarity.
    Returns 0.0 on embedding failure.
    """
    try:
        original_sentences = nltk.sent_tokenize(original_text)
        reconstructed_sentences = nltk.sent_tokenize(reconstructed_text)

        emb_orig_full = embedding_fn(original_text, model_instance)
        emb_recon_full = embedding_fn(reconstructed_text, model_instance)

        # Check for empty arrays from failed Gensim lookup
        if emb_orig_full.size == 0 or emb_recon_full.size == 0:
            logging.warning("Embedding failed for one or both texts (size 0). Similarity is 0.")
            return 0.0

        # Ensure embeddings are 2D for cosine_similarity
        emb_orig_full = emb_orig_full.reshape(1, -1)
        emb_recon_full = emb_recon_full.reshape(1, -1)

        if emb_orig_full.shape[1] != emb_recon_full.shape[1]:
             logging.warning(f"Embedding dimensions mismatch: {emb_orig_full.shape[1]} vs {emb_recon_full.shape[1]}. Cannot compute similarity.")
             return 0.0

        if len(original_sentences) == len(reconstructed_sentences) and len(original_sentences) > 1:
            similarities = []
            for orig_sent, recon_sent in zip(original_sentences, reconstructed_sentences):
                emb_orig_sent = embedding_fn(orig_sent, model_instance)
                emb_recon_sent = embedding_fn(recon_sent, model_instance)

                if emb_orig_sent.size == 0 or emb_recon_sent.size == 0:
                    logging.warning(f"Skipping sentence pair due to embedding failure:\nOrig: {orig_sent[:30]}...\nRecon: {recon_sent[:30]}...")
                    continue # Skip sentence pair if embedding failed

                # Reshape sentence embeddings
                emb_orig_sent = emb_orig_sent.reshape(1, -1)
                emb_recon_sent = emb_recon_sent.reshape(1, -1)

                if emb_orig_sent.shape[1] != emb_recon_sent.shape[1]:
                    logging.warning("Sentence embedding dimensions mismatch.")
                    continue

                sim = cosine_similarity(emb_orig_sent, emb_recon_sent)[0][0]
                similarities.append(sim)

            # Return average sentence similarity, or fallback to full if list is empty
            return np.mean(similarities) if similarities else cosine_similarity(emb_orig_full, emb_recon_full)[0][0]
        else:
            # Fallback to full text comparison
            return cosine_similarity(emb_orig_full, emb_recon_full)[0][0]

    except Exception as e:
        logging.error(f"Error in calculate_cosine_similarity_scores: {e}")
        return 0.0


# --- DELIVERABLE 1: Text Reconstruction ---

# A. Custom Reconstruction of 2+ Sentences
def reconstruct_A_custom(sentences_to_fix: list[str]) -> list[str]:
    """Applies specific, hardcoded fixes to a list of sentences."""
    reconstructed_sentences = []
    for original_sentence in sentences_to_fix:
        reconstructed = original_sentence
        # Apply rules sequentially (order might matter)
        if "Thank your message to show our words to the doctor" in reconstructed:
            reconstructed = reconstructed.replace("Thank your message to show our words to the doctor",
                                                "Thank you for your message conveying our thoughts to the doctor")
        if "as his next contract checking, to all of us" in reconstructed:
            reconstructed = reconstructed.replace("as his next contract checking, to all of us",
                                                "for his review regarding the contract, which concerns all of us")
        if "I am very appreciated" in reconstructed: # Common error pattern
            reconstructed = reconstructed.replace("I am very appreciated", "I very much appreciate")
        if "During our final discuss" in reconstructed:
             reconstructed = reconstructed.replace("During our final discuss", "During our final discussion")
        if "the updates was confusing as it not included" in reconstructed: # Subject-verb agreement + negation
            reconstructed = reconstructed.replace("the updates was confusing as it not included",
                                                "the updates were confusing as they did not include")
        if "although bit delay and less communication at recent days" in reconstructed: # Article and phrasing
            reconstructed = reconstructed.replace("although bit delay and less communication at recent days",
                                                "although there was a bit of delay and less communication in recent days")
        if "they really tried best for paper and cooperation" in reconstructed: # Phrasing
            reconstructed = reconstructed.replace("they really tried best for paper and cooperation",
                                                "they really tried their best on the paper and showed good cooperation")
        reconstructed_sentences.append(reconstructed)
    return reconstructed_sentences

# B. Reconstruction with 3 Python Libraries/Pipelines
def reconstruct_B_with_language_tool(text: str) -> str:
    """Corrects text using LanguageTool."""
    if lang_tool is None:
        logging.warning("LanguageTool not available. Returning original text.")
        return text

    try:
        matches = lang_tool.check(text)
        if matches:
            logging.debug(f"LanguageTool found {len(matches)} potential issues.")


        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
    except Exception as e:
        logging.error(f"Error applying corrections with language_tool: {e}")
        return text 

def reconstruct_B_with_spacy_rules(text: str, nlp_model: spacy.language.Language) -> str:
    """
    Reconstructs text using SpaCy for sentence segmentation and applies
    custom linguistic rules (can be expanded).
    """
    if nlp_model is None:
        logging.warning("SpaCy model not available for rule-based reconstruction. Returning original text.")
        return text
    try:
        doc = nlp_model(text)
        reconstructed_sentences = []

        for sent in doc.sents:
            reconstructed_sentence = sent.text # Start with the sentence text

            # --- Example Rules ---
            # Rule 1: Fix common "very appreciated" error
            if "very appreciated" in reconstructed_sentence:
                reconstructed_sentence = reconstructed_sentence.replace("very appreciated", "very much appreciate")

            # Rule 2: Fix "final discuss" noun error (simple version)
            if "final discuss" in reconstructed_sentence:
                 reconstructed_sentence = reconstructed_sentence.replace("final discuss", "final discussion")

            # Rule 3: Fix "updates was" subject-verb agreement (simple version)
            if "updates was" in reconstructed_sentence:
                 reconstructed_sentence = reconstructed_sentence.replace("updates was", "updates were")

            reconstructed_sentences.append(reconstructed_sentence.strip()) # Strip extra whitespace
        return " ".join(reconstructed_sentences)
    except Exception as e:
        logging.error(f"Error in reconstruct_B_with_spacy_rules: {e}")
        return text

def reconstruct_B_with_pyspellchecker(text: str, spacy_nlp_for_detokenize: spacy.language.Language = None) -> str:
    """Corrects spelling errors using SpellChecker and attempts sensible detokenization."""
    try:
        spell = SpellChecker()
        words = nltk.word_tokenize(text)

        # Find potentially misspelled words
        misspelled = spell.unknown(words)

        corrected_words = []
        for word in words:
            if word in misspelled and word.isalpha(): # Check only potentially misspelled alpha words
                corrected_word = spell.correction(word)
                # Use correction only if it's not None (None means no good suggestion)
                corrected_words.append(corrected_word if corrected_word else word)
            else:
                corrected_words.append(word) # Keep punctuation, numbers, known words

        # Detokenize words back into a string
        detokenizer = TreebankWordDetokenizer()
        temp_text = detokenizer.detokenize(corrected_words)

        if spacy_nlp_for_detokenize:
            doc = spacy_nlp_for_detokenize(temp_text)
            return " ".join([sent.text.strip() for sent in doc.sents])
        else:
             sentences = nltk.sent_tokenize(temp_text)
             return " ".join(sentences) # Simple join

    except Exception as e:
        logging.error(f"Error in reconstruct_B_with_pyspellchecker: {e}")
        return text

# --- Bonus Helper ---
def analyze_mask_context_spacy(text_with_placeholder: str, nlp_model_gr: spacy.language.Language) -> str:
    """Provides basic syntactic context around a placeholder (e.g., MASK_PLACEHOLDER_GR)."""
    if not nlp_model_gr:
        logging.warning("Greek spaCy model not available for syntactic analysis.")
        return "Syntactic analysis skipped."
    if MASK_PLACEHOLDER_GR not in text_with_placeholder:
         return f"Placeholder '{MASK_PLACEHOLDER_GR}' not found in text."

    try:
        doc = nlp_model_gr(text_with_placeholder)
        analysis_str = ""
        found_mask = False
        for token in doc:
            if token.text == MASK_PLACEHOLDER_GR:
                found_mask = True
                head = token.head
                dep = token.dep_
                analysis_str += f"  - Placeholder '{MASK_PLACEHOLDER_GR}': Head='{head.text}' ({head.pos_}), Dep='{dep}'"
                # Get a small window of context tokens
                context_window = 3
                start = max(0, token.i - context_window)
                end = min(len(doc), token.i + context_window + 1)
                context_tokens = [f"{t.text}({t.pos_})" for t in doc[start:end]] # Add POS tag
                analysis_str += f" | Context: ...{' '.join(context_tokens)}...\n"

        if not found_mask: # Should not happen if check above works, but for safety
            return f"Placeholder '{MASK_PLACEHOLDER_GR}' not found during token iteration."
        return analysis_str.strip() if analysis_str else "Analysis could not be performed."

    except Exception as e:
        logging.error(f"Error during SpaCy Greek analysis: {e}")
        return "Syntactic analysis failed."

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting NLP Assignment Execution ---")

    # Download NLTK resources first
    download_nltk_resources()

    # Load models
    load_models() # Populate global model variables

    # --- DELIVERABLE 1: Text Reconstruction ---
    logging.info("\n--- DELIVERABLE 1: Text Reconstruction ---")

    # A. Custom Reconstruction
    logging.info("\nPart A: Custom Reconstruction of Selected Sentences")
    chosen_sentences_for_A = [
        # Sentences chosen to demonstrate specific fixes
        "Thank your message to show our words to the doctor, as his next contract checking, to all of us.", # From Text 1
        "I am very appreciated the full support of the professor, for our Springer proceedings publication", # From Text 1
        "During our final discuss, I told him about the new submission – the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?", # From Text 2
        "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation." # From Text 2
    ]

    logging.info("\nOriginal Sentences for Part A:")
    for s in chosen_sentences_for_A: logging.info(f"- {s}")

    reconstructed_A_sentences = reconstruct_A_custom(chosen_sentences_for_A)
    logging.info("\nReconstructed Sentences (Part A - Custom):")
    for orig, recon in zip(chosen_sentences_for_A, reconstructed_A_sentences):
        logging.info(f"Original: {orig}\nReconstr: {recon}\n")

    # Generate full texts with custom fixes applied for Deliverable 2 comparison
    custom_A_full_text1 = TEXT_1
    custom_A_full_text2 = TEXT_2
    if len(reconstructed_A_sentences) >= 2:
         custom_A_full_text1 = TEXT_1.replace(chosen_sentences_for_A[0], reconstructed_A_sentences[0])
         custom_A_full_text1 = custom_A_full_text1.replace(chosen_sentences_for_A[1], reconstructed_A_sentences[1])
    if len(reconstructed_A_sentences) >= 4:
         custom_A_full_text2 = TEXT_2.replace(chosen_sentences_for_A[2], reconstructed_A_sentences[2])
         custom_A_full_text2 = custom_A_full_text2.replace(chosen_sentences_for_A[3], reconstructed_A_sentences[3])
    logging.info("Applied custom fixes to generate 'Custom_A_Full' versions for analysis.")

    # B. Reconstruction with 3 Python Libraries
    logging.info("\nPart B: Reconstruction of Full Texts with 3 Libraries")

    logging.info("\nReconstruction with LanguageTool...")
    reconstructed_B1_text1 = reconstruct_B_with_language_tool(TEXT_1)
    reconstructed_B1_text2 = reconstruct_B_with_language_tool(TEXT_2)
    logging.info(f"Original Text 1 Length: {len(TEXT_1)}, Reconstructed (LangTool) Length: {len(reconstructed_B1_text1)}")

    logging.info(f"Original Text 2 Length: {len(TEXT_2)}, Reconstructed (LangTool) Length: {len(reconstructed_B1_text2)}")

    logging.info("\nReconstruction with spaCy + Custom Rules...")
    reconstructed_B2_text1 = reconstruct_B_with_spacy_rules(TEXT_1, nlp_spacy)
    reconstructed_B2_text2 = reconstruct_B_with_spacy_rules(TEXT_2, nlp_spacy)
    logging.info(f"Original Text 1 Length: {len(TEXT_1)}, Reconstructed (SpaCy Rules) Length: {len(reconstructed_B2_text1)}")
    logging.info(f"Original Text 2 Length: {len(TEXT_2)}, Reconstructed (SpaCy Rules) Length: {len(reconstructed_B2_text2)}")

    logging.info("\nReconstruction with Pyspellchecker (Spelling Focus)...")
    reconstructed_B3_text1 = reconstruct_B_with_pyspellchecker(TEXT_1, nlp_spacy)
    reconstructed_B3_text2 = reconstruct_B_with_pyspellchecker(TEXT_2, nlp_spacy)
    logging.info(f"Original Text 1 Length: {len(TEXT_1)}, Reconstructed (Pyspell) Length: {len(reconstructed_B3_text1)}")
    logging.info(f"Original Text 2 Length: {len(TEXT_2)}, Reconstructed (Pyspell) Length: {len(reconstructed_B3_text2)}")

    # Store reconstructions for Deliverable 2
    reconstructions_text1 = {
        "Original": TEXT_1,
        "Custom_A_Full": custom_A_full_text1,
        "Lib_LanguageTool": reconstructed_B1_text1,
        "Lib_SpaCyRules": reconstructed_B2_text1,
        "Lib_Pyspellchecker": reconstructed_B3_text1
    }
    reconstructions_text2 = {
        "Original": TEXT_2,
        "Custom_A_Full": custom_A_full_text2,
        "Lib_LanguageTool": reconstructed_B1_text2,
        "Lib_SpaCyRules": reconstructed_B2_text2,
        "Lib_Pyspellchecker": reconstructed_B3_text2
    }

    # --- DELIVERABLE 2: Computational Analysis ---
    logging.info("\n\n--- DELIVERABLE 2: Computational Analysis ---")

    embedding_functions = {}
    if sbert_model:
        embedding_functions["SentenceBERT"] = (get_sentence_embedding_sbert, sbert_model)
    if nlp_spacy and ACTUAL_SPACY_DIM > 0:
        embedding_functions["SpaCy DocVec"] = (get_sentence_embedding_spacy, nlp_spacy)
    if word2vec_model:
        embedding_functions["Word2Vec Avg"] = (get_sentence_embedding_gensim_avg, word2vec_model)
    if fasttext_model:
        embedding_functions["FastText Avg"] = (get_sentence_embedding_gensim_avg, fasttext_model)

    # Check if any embedding functions loaded
    if not embedding_functions:
         logging.error("No embedding models loaded successfully. Cannot perform Deliverable 2 Analysis.")
    else:
        all_texts_for_vis = []
        similarity_results = [] # For potential pandas DataFrame later

        logging.info("\nCosine Similarity Analysis (Original vs. Reconstructed):")
        for text_id, reconstructions in [("Text 1", reconstructions_text1), ("Text 2", reconstructions_text2)]:
            logging.info(f"\n--- {text_id} ---")
            original_text = reconstructions["Original"]
            all_texts_for_vis.append({"text": original_text, "label": f"{text_id}_Original", "type": "Original", "source": text_id})

            for recon_name, recon_text in reconstructions.items():
                if recon_name == "Original":
                    continue

                all_texts_for_vis.append({"text": recon_text, "label": f"{text_id}_{recon_name}", "type": recon_name, "source": text_id})
                logging.info(f"  Comparing Original with '{recon_name}':")

                for emb_name, (emb_fn, model_instance) in embedding_functions.items():
                    similarity = calculate_cosine_similarity_scores(original_text, recon_text, emb_fn, model_instance)
                    logging.info(f"    - Cosine Similarity ({emb_name}): {similarity:.4f}")
                    similarity_results.append({
                        "TextID": text_id,
                        "Reconstruction": recon_name,
                        "Embedding": emb_name,
                        "Similarity": similarity
                    })
                logging.info("-" * 20)
        df_similarity = pd.DataFrame(similarity_results)
        logging.info("\nSimilarity Results Summary:\n" + df_similarity.to_string())

        # --- Visualization ---
        if sbert_model and all_texts_for_vis:
            logging.info("\nVisualizing Embeddings with PCA and t-SNE (using SentenceBERT)...")

            labels_for_plot = [item["label"] for item in all_texts_for_vis]
            text_contents_for_plot = [item["text"] for item in all_texts_for_vis]
            types_for_plot = [item["type"] for item in all_texts_for_vis] # Original, Custom_A_Full, etc.
            source_for_plot = [item["source"] for item in all_texts_for_vis] # Text 1, Text 2

            # Use the SBERT embedding function
            sbert_emb_fn, sbert_inst = embedding_functions["SentenceBERT"]
            embeddings_for_plot = np.array([sbert_emb_fn(text, sbert_inst) for text in text_contents_for_plot])

            # Filter out any zero-sum embeddings
            valid_indices = [i for i, emb in enumerate(embeddings_for_plot) if np.sum(np.abs(emb)) > 1e-6]
            if len(valid_indices) < len(embeddings_for_plot):
                logging.warning(f"{len(embeddings_for_plot) - len(valid_indices)} texts had near-zero embeddings and were excluded from visualization.")
                embeddings_for_plot = embeddings_for_plot[valid_indices]
                labels_for_plot = [labels_for_plot[i] for i in valid_indices]
                types_for_plot = [types_for_plot[i] for i in valid_indices]
                source_for_plot = [source_for_plot[i] for i in valid_indices]

            if len(embeddings_for_plot) < 2:
                logging.warning("Not enough valid embeddings (less than 2) to perform PCA/t-SNE visualization.")
            else:
                # PCA
                pca = PCA(n_components=2, random_state=42)
                embeddings_pca = pca.fit_transform(embeddings_for_plot)

                plt.figure(figsize=(18, 8)) # Adjusted figure size
                plt.subplot(1, 2, 1)
                sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1],
                                hue=types_for_plot, style=source_for_plot, s=100, palette="viridis")
                plt.title('PCA of Text Embeddings (SentenceBERT)')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                # Adjust legend position
                plt.legend(title='Type / Source', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

                # t-SNE
                n_samples = len(embeddings_for_plot)
                # Robust perplexity setting
                perplexity_val = min(30.0, float(max(1, n_samples - 1)))

                if n_samples > 1:
                    try:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val,
                                    n_iter=350, # Slightly more iterations
                                    learning_rate='auto', init='pca')
                        embeddings_tsne = tsne.fit_transform(embeddings_for_plot)

                        plt.subplot(1, 2, 2)
                        sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1],
                                        hue=types_for_plot, style=source_for_plot, s=100, palette="viridis")
                        plt.title('t-SNE of Text Embeddings (SentenceBERT)')
                        plt.xlabel('t-SNE Component 1')
                        plt.ylabel('t-SNE Component 2')
                        plt.legend(title='Type / Source', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                    except ValueError as ve:
                         logging.error(f"t-SNE failed: {ve}. Might be due to perplexity ({perplexity_val}) vs n_samples ({n_samples}).")
                    except Exception as e:
                         logging.error(f"An unexpected error occurred during t-SNE: {e}")
                else:
                    logging.warning("Skipping t-SNE: Need at least 2 samples.")

                plt.suptitle("Embedding Visualizations (SentenceBERT)", fontsize=16, y=0.99)
                plt.tight_layout(rect=[0, 0, 0.88, 0.97])
                plt.show()
        else:
            logging.warning("Skipping visualization: SBERT model not loaded or no texts to visualize.")

    # --- BONUS: Masked Clause Input (Greek) ---
    logging.info("\n\n--- BONUS: Masked Clause Input (Greek) ---")

    if fill_mask_pipeline_greek:
        masked_texts_greek = [
            ("Άρθρο 1113: Κοινό πράγμα.", # Context Key
             f"Αν η κυριότητα του {fill_mask_pipeline_greek.tokenizer.mask_token} ανήκει σε περισσότερους {fill_mask_pipeline_greek.tokenizer.mask_token} αδιαιρέτου κατ΄ ιδανικά {fill_mask_pipeline_greek.tokenizer.mask_token}, εφαρμόζονται οι διατάξεις για την κοινωνία."), # Text
            ("Άρθρο 1114: Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου.",
             f"Στο κοινό {fill_mask_pipeline_greek.tokenizer.mask_token} μπορεί να συσταθεί πραγματική δουλεία υπέρ του {fill_mask_pipeline_greek.tokenizer.mask_token} κύριου άλλου ακινήτου και αν ακόμη αυτός είναι {fill_mask_pipeline_greek.tokenizer.mask_token} του ακινήτου που βαρύνεται με τη δουλεία. Το ίδιο ισχύει και για την {fill_mask_pipeline_greek.tokenizer.mask_token} δουλεία πάνω σε ακίνητο υπέρ των εκάστοτε κυρίων κοινού ακινήτου, αν {fill_mask_pipeline_greek.tokenizer.mask_token} από αυτούς είναι κύριος του {fill_mask_pipeline_greek.tokenizer.mask_token} που βαρύνεται με τη δουλεία.")
        ]

        # More structured ground truth corresponding to mask order
        ground_truths_greek_structured = {
             "Άρθρο 1113: Κοινό πράγμα.": [
                 "πράγματος",   # Mask 1
                 "συγκοινωνούς", # Mask 2
                 "μέρη"         # Mask 3 
             ],
             "Άρθρο 1114: Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου.": [
                 "ακίνητο",   
                 "κυρίου",     
                 "συγκύριος", 
                 "πραγματική", 
                 "ένας",      
                 "ακινήτου"    
             ]
        }

        for context_key, text_gr_masked in masked_texts_greek:
            logging.info(f"\nAnalyzing: {context_key}")
            logging.info(f"Masked Text: {text_gr_masked}")

            syntactic_analysis_hints = analyze_mask_context_spacy(
                text_gr_masked.replace(fill_mask_pipeline_greek.tokenizer.mask_token, MASK_PLACEHOLDER_GR),
                nlp_greek
            )
            logging.info(f"  Syntactic Context Analysis Hints:\n{syntactic_analysis_hints}")

            try:
                predictions_sets = fill_mask_pipeline_greek(text_gr_masked)
            except Exception as e:
                logging.error(f"Fill-mask pipeline failed for '{context_key}': {e}")
                continue 

            if not isinstance(predictions_sets, list): predictions_sets = [] # Error case
            if len(predictions_sets) > 0 and not isinstance(predictions_sets[0], list): # Single mask case
                 predictions_sets = [predictions_sets]

            current_gt_words = ground_truths_greek_structured.get(context_key, [])
            num_masks_in_text = text_gr_masked.count(fill_mask_pipeline_greek.tokenizer.mask_token)

            if len(predictions_sets) != num_masks_in_text:
                 logging.warning(f"Mismatch between predicted mask sets ({len(predictions_sets)}) and masks in text ({num_masks_in_text}) for '{context_key}'. Skipping detailed comparison.")
                 continue

            for i, pred_set in enumerate(predictions_sets):
                mask_num = i + 1
                logging.info(f"  Predictions for MASK #{mask_num}:")
                gt_word_for_mask = current_gt_words[i] if i < len(current_gt_words) else "N/A (GT missing)"
                logging.info(f"    (Ground Truth for this mask: {gt_word_for_mask})")

                found_gt_in_top_k = False
                if not isinstance(pred_set, list): # Check if prediction set is valid
                     logging.warning(f"    Prediction set for mask {mask_num} is not a list. Skipping.")
                     continue

                top_k = len(pred_set)
                for prediction in pred_set: # pred_set should be a list of dicts
                     if isinstance(prediction, dict) and 'token_str' in prediction and 'score' in prediction:
                         token_str = prediction['token_str'].strip() # Clean whitespace
                         score = prediction['score']
                         logging.info(f"    - Token: '{token_str}' (Score: {score:.4f})")
                         if token_str == gt_word_for_mask:
                             found_gt_in_top_k = True
                     else:
                          logging.warning(f"    Unexpected prediction format: {prediction}")

                if gt_word_for_mask != "N/A (GT missing)":
                    if found_gt_in_top_k:
                        logging.info(f"    -> Ground truth '{gt_word_for_mask}' was in top {top_k} predictions.")
                    else:
                        logging.info(f"    -> Ground truth '{gt_word_for_mask}' was NOT in top {top_k} predictions.")
            logging.info("-" * 30)

    else:
        logging.warning("Skipping Greek masked clause input: Fill-mask pipeline not loaded.")

    logging.info("\n--- NLP Assignment Execution Finished ---")