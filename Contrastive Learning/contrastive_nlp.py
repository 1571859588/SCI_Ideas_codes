import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
import deepspeed
from unsloth import FastLanguageModel
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ContrastiveLearningConfig:
    temperature: float = 0.07
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    model_name: str = "microsoft/deberta-v3-base"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

class ContrastiveNLPModel(nn.Module):
    def __init__(self, config: ContrastiveLearningConfig):
        super().__init__()
        self.config = config
        
        # Initialize base model with Unsloth optimization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length,
            device="auto",
        )
        
        # Add LoRA adapters
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Projection head for contrastive learning
        hidden_size = self.model.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)  # Final embedding dimension
        )

    def encode_text(self, text: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model(**inputs)
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0]
        # Project embeddings
        projected = self.projection(embeddings)
        return F.normalize(projected, p=2, dim=1)

    def forward(
        self,
        anchor_texts: List[str],
        positive_texts: List[str],
        negative_texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode all texts
        anchor_embeddings = self.encode_text(anchor_texts)
        positive_embeddings = self.encode_text(positive_texts)
        negative_embeddings = self.encode_text(negative_texts)
        
        return anchor_embeddings, positive_embeddings, negative_embeddings

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
        
        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)

class EDAContrastiveDataset:
    def __init__(
        self,
        concepts: List[str],
        descriptions: List[str],
        negative_generator: callable
    ):
        self.concepts = concepts
        self.descriptions = descriptions
        self.negative_generator = negative_generator

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        concept = self.concepts[idx]
        description = self.descriptions[idx]
        negative = self.negative_generator(concept, description)
        
        return {
            "anchor": concept,
            "positive": description,
            "negative": negative
        }

def train_model(
    model: ContrastiveNLPModel,
    train_dataset: EDAContrastiveDataset,
    config: ContrastiveLearningConfig
):
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": config.learning_rate
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2
            }
        }
    )
    
    criterion = ContrastiveLoss(temperature=config.temperature)
    
    for epoch in range(config.num_epochs):
        model_engine.train()
        total_loss = 0
        
        for i in range(0, len(train_dataset), config.batch_size):
            batch = [train_dataset[j] for j in range(i, min(i + config.batch_size, len(train_dataset)))]
            
            anchor_texts = [item["anchor"] for item in batch]
            positive_texts = [item["positive"] for item in batch]
            negative_texts = [item["negative"] for item in batch]
            
            anchor_emb, positive_emb, negative_emb = model_engine(
                anchor_texts, positive_texts, negative_texts
            )
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_dataset) // config.batch_size)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}")

# Example usage:
def create_negative_sample(concept: str, description: str) -> str:
    """
    Create a negative sample by replacing key terms or modifying the description.
    This is a simple example - in practice, you'd want a more sophisticated approach.
    """
    # This is a placeholder - implement your domain-specific negative sample generation
    return description.replace(concept, "different_concept")

if __name__ == "__main__":
    # Example data
    concepts = ["transistor", "capacitor", "resistor"]
    descriptions = [
        "A semiconductor device used to amplify or switch electronic signals",
        "A passive electronic component that stores electrical energy in an electric field",
        "A passive electronic component that implements electrical resistance"
    ]
    
    # Create dataset
    dataset = EDAContrastiveDataset(
        concepts=concepts,
        descriptions=descriptions,
        negative_generator=create_negative_sample
    )
    
    # Initialize model and config
    config = ContrastiveLearningConfig()
    model = ContrastiveNLPModel(config)
    
    # Train model
    train_model(model, dataset, config) 