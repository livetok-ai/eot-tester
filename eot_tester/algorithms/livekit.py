"""LiveKit-based EOT algorithm implementation."""

import asyncio
import json
import logging
import math
import re
import unicodedata
from typing import Optional, Dict, Any, List

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download, errors
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from ..base import EOTAlgorithm, EOTResult, EOTState


class LiveKitAlgorithm(EOTAlgorithm):
    """
    LiveKit-based end-of-turn detection algorithm.
    
    Uses ONNX-optimized models from HuggingFace (livekit/turn-detector)
    to predict end-of-utterance probability based on conversation context.
    
    Supports both English-only and multilingual models with contextual
    awareness and language-specific thresholds.
    """
    
    HG_MODEL = "livekit/turn-detector"
    ONNX_FILENAME = "model_q8.onnx"
    MODEL_REVISIONS = {
        "en": "v1.2.2-en",
        "multilingual": "v0.3.0-intl",
    }
    MAX_HISTORY_TOKENS = 128
    MAX_HISTORY_TURNS = 6
    
    def __init__(self, model_type: str = "en", language: Optional[str] = "en"):
        """
        Initialize LiveKit algorithm.
        
        Args:
            model_type: Either 'en' for English-only or 'multilingual'
            unlikely_threshold: Custom threshold override (not recommended)
            language: Language code for language-specific threshold (e.g., 'en', 'es', 'fr')
        """
        super().__init__(f"LiveKit-{model_type}")
        self.model_type = model_type
        self.language = language
        self.session = None
        self.tokenizer = None
        self.languages = {}
        self.logger = logging.getLogger(__name__)
        
        if self.model_type not in self.MODEL_REVISIONS:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def initialize(self) -> None:
        """Initialize the LiveKit model and tokenizer."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required dependencies not available. Install with: "
                "pip install onnxruntime transformers huggingface_hub"
            )
        
        model_revision = self.MODEL_REVISIONS[self.model_type]
        
        try:
            # Download ONNX model
            self.logger.info(f"Loading LiveKit model: {self.HG_MODEL} (revision: {model_revision})")
            local_path_onnx = hf_hub_download(
                repo_id=self.HG_MODEL,
                filename=self.ONNX_FILENAME,
                subfolder="onnx",
                revision=model_revision,
            )
            
            # Set up ONNX session with CPU optimizations
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = max(1, math.ceil(4) // 2)  # Conservative CPU usage
            sess_options.inter_op_num_threads = 1
            sess_options.add_session_config_entry("session.dynamic_block_base", "4")
            
            self.session = ort.InferenceSession(
                local_path_onnx, 
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.HG_MODEL,
                revision=model_revision,
                truncation_side="left",
            )
            
            # Load language configuration
            try:
                config_path = hf_hub_download(
                    repo_id=self.HG_MODEL,
                    filename="languages.json",
                    revision=model_revision,
                )
                with open(config_path, 'r') as f:
                    self.languages = json.load(f)
                self.logger.info(f"Loaded language support for {len(self.languages)} languages")
            except Exception as e:
                self.logger.warning(f"Could not load language config: {e}")
                self.languages = {}
            
            self.logger.info("LiveKit model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load LiveKit model: {e}")
            raise
    
    def detect(self, text: str, context: Optional[str] = None) -> EOTResult:
        """
        Detect end-of-turn state using LiveKit algorithm.
        
        Args:
            text: Input text to analyze
            context: Optional conversation context or system prompt
            
        Returns:
            EOTResult with detected state and confidence
        """
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not text.strip():
            return EOTResult(EOTState.UNFINISHED, confidence=0.5)
        
        try:
            # Build conversation context
            chat_context = self._build_chat_context(text, context)
            
            # Get end-of-utterance probability
            eou_probability = self._predict_eou_probability(chat_context)
            
            # Convert probability to EOT state using language-specific threshold
            state = self._probability_to_state(eou_probability, self.language)
            
            return EOTResult(
                state=state,
                confidence=eou_probability,
                metadata={
                    "eou_probability": eou_probability,
                    "model_type": self.model_type,
                    "language": self.language,
                    "threshold": self._get_language_threshold(self.language),
                    "context_turns": len(chat_context)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error during LiveKit detection: {e}")
            return EOTResult(EOTState.UNFINISHED, confidence=0.0)
    
    def _build_chat_context(self, text: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Build chat context from input text and optional context."""
        messages = []
        
        # Add system context if provided
        if context:
            messages.append({"role": "assistant", "content": context})
        
        # Add current user input
        messages.append({"role": "user", "content": text})
        
        # Limit to max history turns
        return messages[-self.MAX_HISTORY_TURNS:]
    
    def _predict_eou_probability(self, chat_context: List[Dict[str, str]]) -> float:
        """Predict end-of-utterance probability using ONNX model."""
        # Format chat context for the model
        formatted_text = self._format_chat_context(chat_context)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=self.MAX_HISTORY_TOKENS,
            truncation=True,
        )
        
        # Run ONNX inference
        outputs = self.session.run(
            None, 
            {"input_ids": inputs["input_ids"].astype("int64")}
        )
        
        # Extract probability from output
        eou_probability = float(outputs[0].flatten()[-1])
        return eou_probability
    
    def _format_chat_context(self, chat_context: List[Dict[str, str]]) -> str:
        """Format chat context using tokenizer's chat template."""
        # Normalize and combine adjacent messages from same role
        normalized_context = []
        last_msg = None
        
        for msg in chat_context:
            if not msg["content"]:
                continue
            
            content = self._normalize_text(msg["content"])
            
            # Combine adjacent turns from same role (matching training data)
            if last_msg and last_msg["role"] == msg["role"]:
                last_msg["content"] += f" {content}"
            else:
                normalized_msg = {"role": msg["role"], "content": content}
                normalized_context.append(normalized_msg)
                last_msg = normalized_msg
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            normalized_context,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )
        
        # Remove the EOU token from current utterance (as per LiveKit implementation)
        if isinstance(formatted_text, str):
            ix = formatted_text.rfind("<|im_end|>")
            if ix != -1:
                formatted_text = formatted_text[:ix]
        
        return formatted_text or ""
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for the model (different strategies for EN vs multilingual)."""
        if not text:
            return ""
        
        if self.model_type == "en":
            # English model uses original text without normalization
            return text
        else:
            # Multilingual model uses Unicode normalization
            text = unicodedata.normalize("NFKC", text.lower())
            text = "".join(
                ch for ch in text
                if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
            )
            text = re.sub(r"\s+", " ", text).strip()
            return text
    
    def _probability_to_state(self, probability: float, language: Optional[str] = None) -> EOTState:
        """Convert end-of-utterance probability to EOT state."""
        print(probability, language)
        # Get language-specific threshold from languages.json
        threshold = self._get_language_threshold(language)
        
        if probability >= threshold:
            return EOTState.FINISHED
        else:
            return EOTState.UNFINISHED
    
    def _get_language_threshold(self, language: Optional[str] = None) -> float:
        """Get the threshold for a specific language from languages.json."""
        
        # If no language specified or no languages loaded, use default
        if not language or not self.languages:
            return 0.5
        
        lang = language.lower()
        lang_data = self.languages.get(lang)
        
        # Try base language if full language code not found (e.g. 'en-US' -> 'en')
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self.languages.get(base_lang)
        
        if lang_data and "threshold" in lang_data:
            return float(lang_data["threshold"])
        
        # Default threshold if language not found
        return 0.5
    
    def supports_language(self, language: Optional[str]) -> bool:
        """Check if the given language is supported."""
        if not language or not self.languages:
            return True  # Assume supported if no language info
        
        lang = language.lower()
        # Try full language code first
        if lang in self.languages:
            return True
        
        # Try base language if full code not found
        if "-" in lang:
            base_lang = lang.split("-")[0]
            return base_lang in self.languages
        
        return False
    
    def get_language_threshold(self, language: Optional[str]) -> Optional[float]:
        """Get the threshold for a specific language."""
        if not language or not self.languages:
            return None
        
        lang = language.lower()
        lang_data = self.languages.get(lang)
        
        # Try base language if full language code not found
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self.languages.get(base_lang)
        
        return lang_data.get("threshold") if lang_data else None