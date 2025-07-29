"""TEN (Turn Detection) algorithm implementation."""

import logging
from typing import Optional, Dict, Any

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..base import EOTAlgorithm, EOTResult, EOTState


class TENAlgorithm(EOTAlgorithm):
    """
    TEN Turn Detection algorithm based on Qwen2.5-7B model.
    
    Classifies text into three states:
    - finished: Complete thought expecting a response
    - wait: User instructing AI not to speak  
    - unfinished: User momentarily paused but intends to continue
    """
    
    MODEL_ID = "TEN-framework/TEN_Turn_Detection"
    
    def __init__(self):
        super().__init__("TEN")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the TEN model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for TEN algorithm. "
                "Install with: pip install transformers torch"
            )
        
        try:
            self.logger.info(f"Loading TEN model: {self.MODEL_ID}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info("TEN model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load TEN model: {e}")
            raise
    
    def detect(self, text: str, context: Optional[str] = None) -> EOTResult:
        """
        Detect end-of-turn state using TEN algorithm.
        
        Args:
            text: Input text to analyze
            context: Optional system prompt/context
            
        Returns:
            EOTResult with detected state
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        if not text.strip():
            return EOTResult(EOTState.UNFINISHED, confidence=0.5)
        
        try:
            # Prepare messages for chat template
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": text})
            
            # Apply chat template and tokenize
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip().lower()
            
            # Parse result to EOT state
            state = self._parse_result(generated_text)
            confidence = self._calculate_confidence(generated_text)
            
            return EOTResult(
                state=state, 
                confidence=confidence,
                metadata={"raw_output": generated_text}
            )
            
        except Exception as e:
            self.logger.error(f"Error during TEN detection: {e}")
            return EOTResult(EOTState.UNFINISHED, confidence=0.0)
    
    def _parse_result(self, generated_text: str) -> EOTState:
        """Parse model output to determine EOT state."""
        text_lower = generated_text.lower()
        
        if "unfinished" in text_lower:
            return EOTState.UNFINISHED
        elif "finished" in text_lower:
            return EOTState.FINISHED
        elif "wait" in text_lower:
            return EOTState.WAIT
        else:
            # Default fallback based on text characteristics
            if text_lower.endswith(('.', '!', '?')):
                return EOTState.FINISHED
            else:
                return EOTState.UNFINISHED
    
    def _calculate_confidence(self, generated_text: str) -> float:
        """Calculate confidence score based on model output clarity."""
        text_lower = generated_text.lower()
        
        # High confidence if exact match
        if text_lower in ["finished", "wait", "unfinished"]:
            return 1.0
        
        # Medium confidence if contains keywords
        keywords = ["finished", "wait", "unfinished"]
        for keyword in keywords:
            if keyword in text_lower:
                return 0.8
        
        # Low confidence otherwise
        return 0.3