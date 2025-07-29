"""Pipecat Smart Turn V2 algorithm implementation."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torchaudio
from openai import OpenAI
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel, Wav2Vec2Model
from torch import nn
import torch.nn.functional as F

from ..base import EOTAlgorithm, EOTResult, EOTState


class Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    """Custom Wav2Vec2 model for turn completion detection."""
    
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Weight initialization for classifier and pool_attention layers
        for module in list(self.classifier) + list(self.pool_attention):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        """Apply attention pooling to hidden states."""
        # Calculate attention weights
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to hidden states
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)

        return weighted_sum

    def forward(self, input_values, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        # Create transformer padding mask
        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)

        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            # Calculate positive sample weight based on batch statistics
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            # Add L2 regularization for classifier layers
            l2_lambda = 0.01
            l2_reg = torch.tensor(0., device=logits.device)
            for param in self.classifier.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}


class PipecatAlgorithm(EOTAlgorithm):
    """Pipecat Smart Turn V2 algorithm for end-of-turn detection."""
    
    def __init__(self, name: str = "pipecat"):
        super().__init__(name)
        self.model = None
        self.processor = None
        self.openai_client = None
        self.device = None
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def initialize(self) -> None:
        """Initialize the Pipecat model and OpenAI client."""
        try:
            MODEL_PATH = "pipecat-ai/smart-turn-v2"
            
            # Load model and processor
            self.model = Wav2Vec2ForEndpointing.from_pretrained(MODEL_PATH)
            self.processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
            
            # Determine device (CPU, MPS for Apple Silicon, or CUDA)
            self.device = "cpu"
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize OpenAI client
            self.openai_client = OpenAI()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pipecat algorithm: {e}")
    
    def _text_to_speech(self, text: str) -> Path:
        """Convert text to speech using OpenAI TTS API with caching."""
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = self.cache_dir / f"tts_{text_hash}.wav"
        
        # Return cached file if it exists
        if cache_file.exists():
            return cache_file
        
        try:
            # Generate speech using OpenAI TTS
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav"
            )
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            return cache_file
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {e}")
    
    def _preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio file - 16kHz numpy array."""
        try:
            # Load audio file at 16kHz
            audio, sr = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
            
            # Convert to numpy and take first channel if stereo
            audio = audio.squeeze().numpy()
            if audio.ndim > 1:
                audio = audio[0]
            
            return audio
            
        except Exception as e:
            raise RuntimeError(f"Failed to process audio: {e}")
    
    def predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.
        This matches the official implementation exactly.

        Args:
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary containing prediction results:
            - prediction: 1 for complete, 0 for incomplete  
            - probability: Probability of completion (sigmoid output)
        """
        # Process audio exactly as in official implementation
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,  # 16 seconds at 16kHz as specified in training
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # The model returns sigmoid probabilities directly in the logits field
        probability = outputs["logits"][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability,
        }
    
    def detect(self, text: str, context: Optional[str] = None) -> EOTResult:
        """
        Detect end-of-turn state using Pipecat Smart Turn V2 model.
        
        Args:
            text: The input text to analyze
            context: Optional context (not used in this implementation)
            
        Returns:
            EOTResult with detected state and confidence
        """
        if self.model is None or self.processor is None or self.openai_client is None:
            raise RuntimeError("Algorithm not initialized. Call initialize() first.")
        
        try:
            # Convert text to speech
            audio_path = self._text_to_speech(text)
            
            # Load and preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Use the official prediction method
            result = self.predict_endpoint(audio)
            
            prediction = result["prediction"]
            probability = result["probability"]
            
            # Map binary prediction to EOT states
            # 1 = Complete turn, 0 = Incomplete turn
            if prediction == 1:
                state = EOTState.FINISHED
            else:
                state = EOTState.UNFINISHED
            
            metadata = {
                'audio_path': str(audio_path),
                'sample_rate': 16000,
                'device': str(self.device),
                'prediction': prediction,
                'probability': probability
            }
            
            return EOTResult(state=state, confidence=probability, metadata=metadata)
            
        except Exception as e:
            # Return default state with low confidence on error
            return EOTResult(
                state=EOTState.UNFINISHED, 
                confidence=0.0, 
                metadata={'error': str(e)}
            )