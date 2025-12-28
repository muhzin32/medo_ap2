import logging
import nltk
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import ssl

# SSL Fix for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK corpora (idempotent)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    logger.info("Downloading NLTK resources...")
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {e}")

def simple_regex_fallback(text):
    """Fallback if NLTK fails: detects 'um', 'uh' and repeated words."""
    clean_tokens = []
    fillers = []
    
    # Simple tokenization by space
    tokens = text.split()
    last_word = None
    
    for word in tokens:
        lower = word.lower().strip(",.!?")
        is_filler = False
        
        # 1. Hardcoded fallback list (since dynamic failed)
        if lower in ['um', 'uh', 'hmm', 'ah', 'er']:
            is_filler = True
            
        # 2. Repetition
        elif last_word and last_word == lower and lower.isalpha():
            is_filler = True
            
        if is_filler:
            fillers.append(word)
        else:
            clean_tokens.append(word)
            last_word = lower
            
    return clean_tokens, fillers

def detect_fillers(text):
    """
    Detects filler words using NLTK POS tagging and repetition patterns.
    Returns the processed text and a list of detected fillers.
    """
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        
        fillers = []
        clean_tokens = []
        
        # Simple repetition state
        last_token = None
        
        for word, tag in tagged:
            term = word.lower()
            is_filler = False
            
            # 1. Linguistic Heuristic: POS Tag 'UH'
            if tag == 'UH':
                is_filler = True
                
            # 2. Linguistic Heuristic: Repetitions
            elif last_token and last_token == term and term.isalpha():
                is_filler = True
                
            if is_filler:
                fillers.append(word)
            else:
                clean_tokens.append(word)
                last_token = term

        return clean_tokens, fillers
        
    except Exception as e:
        logger.error(f"NLTK Detection Failed: {e}. Using regex fallback.")
        return simple_regex_fallback(text)

# Common Devanagari -> English mappings for command words
SCRIPT_CORRECTION_MAP = {
    "स्टॉप": "stop",
    "रुको": "stop",
    "हेलो": "hello",
    "हाय": "hi",
    "ओके": "ok",
    "ठीक": "ok",
    "हैलो": "hello",
    "गुडबाय": "goodbye",
    "बाय": "bye",
    "येस": "yes",
    "हां": "yes",
    "नो": "no",
    "नहीं": "no",
    "नाम": "name",
    "मेरा": "my",
    "माय": "my",
    "नेम": "name",
    "इज़": "is",
    "इज": "is",
    "अ": "a",
    "ऐ": "a",
    "एन": "an",
    "द": "the",
    "व्हाट": "what",
    "व्हेन": "when",
    "व्हेर": "where",
    "हाउ": "how",
    "यू": "you",
    "आर": "are",
    "एम": "am",
    "आई": "i",
    "प्ले": "play",
    "पॉज़": "pause",
    "कॉल": "call"
}

def apply_script_correction(text, context_lang):
    """
    If context is English but text has Devanagari, try to fix common words.
    """
    if context_lang != 'en-IN':
        return text

    corrected_tokens = []
    tokens = text.split()
    
    normalization_occurred = False
    
    for token in tokens:
        # Check stripping punctuation
        clean = token.strip(".,!?।")
        if clean in SCRIPT_CORRECTION_MAP:
             corrected_tokens.append(SCRIPT_CORRECTION_MAP[clean])
             normalization_occurred = True
        else:
             corrected_tokens.append(token)
             
    if normalization_occurred:
        return " ".join(corrected_tokens)
        
    return text

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    config = data.get('config', {})
    context_lang = data.get('language', 'en-IN') # Default to English context
    action = config.get('action', 'remove') # remove, mark, preserve

    if not text:
        return jsonify({'processed_text': '', 'fillers': []})

    # 1. Apply Script Correction FIRST
    # If we are in English mode, fix "स्टॉप" -> "stop"
    text = apply_script_correction(text, context_lang)

    try:
        clean_tokens, fillers = detect_fillers(text)
        
        # Reconstruct text
        # Simple join for fallback/NLTK tokens
        # Note: NLTK tokens include punctuation as separate tokens often, simple join might need spacing fix
        # But for voice usage, readability is secondary to content for LLM. 
        # Ideally use TreebankWordDetokenizer if NLTK available, else space join.
        
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer
            detokenizer = TreebankWordDetokenizer()
            reconstructed = detokenizer.detokenize(clean_tokens)
        except:
            reconstructed = " ".join(clean_tokens)
        
        if action == 'remove':
            processed_text = reconstructed
        elif action == 'mark':
            processed_text = reconstructed
        else:
            processed_text = text

        response = {
            'processed_text': processed_text,
            'fillers_detected': fillers,
            'original_text': text
        }
        
        if fillers:
            logger.info(f"Detected fillers: {fillers}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
