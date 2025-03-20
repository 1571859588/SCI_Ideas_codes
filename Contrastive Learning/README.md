# Contrastive Learning for EDA (Electronic Design Automation)

This repository implements a contrastive learning framework specifically designed for NLP tasks in Electronic Design Automation (EDA). The framework uses state-of-the-art language models and optimization techniques to learn representations of EDA concepts and their descriptions.

## Mathematical Foundation

### Contrastive Loss

The core of our framework is the contrastive loss function, which aims to maximize the similarity between positive pairs while minimizing the similarity between negative pairs.

1. **Similarity Computation**:
   For any two embeddings \(a\) and \(b\), the cosine similarity is computed as:
   $$
   sim(a, b) = \frac{a^T b}{\|a\| \|b\|}
   $$
   
2. **Temperature Scaling**:
   The similarity is scaled by a temperature parameter \($\tau$\):
   $$
   sim_\tau(a, b) = \frac{sim(a, b)}{\tau}
   $$
   
3. **InfoNCE Loss**:
   For an anchor \(a\), positive sample \(p\), and negative samples \($\{n_1, ..., n_k\}$\):
   $$
   \mathcal{L} = -\log \frac{\exp(sim_\tau(a, p))}{\exp(sim_\tau(a, p)) + \sum_{i=1}^k \exp(sim_\tau(a, n_i))}
   $$

### Embedding Space

The model projects EDA concepts into a normalized embedding space where:

1. **Normalization**:
   $$
   \|z\|_2 = 1
   $$
   where \(z\) is the output of the projection head.

2. **Projection Head**:
   $$
   h(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)
   $$
   where \($W_1, W_2$\) are learnable parameters.

### LoRA  Adaptation

Low-Rank Adaptation (LoRA) reduces the number of trainable parameters by:

$$
W = W_0 + BA
$$

where:
- \($W_0$\) is the frozen pretrained weights
- \($B \in \mathbb{R}^{d \times r}$\) and \($A \in \mathbb{R}^{r \times d}$\) are low-rank matrices
- \($r$\) is the LoRA rank (typically \($r \ll d$\))

## Features

- **Advanced Model Architecture**:
  - DeBERTa-v3 base model with LoRA fine-tuning
  - Optimized with Unsloth for faster training
  - DeepSpeed integration for distributed training
  - FP16 mixed precision training

- **Contrastive Learning Components**:
  - Custom contrastive loss implementation
  - Projection head for embedding space optimization
  - Temperature-scaled similarity computation
  - Efficient batch processing

- **EDA-Specific Design**:
  - Specialized for EDA concept learning
  - Customizable negative sample generation
  - Support for circuit and component descriptions
  - Scalable dataset handling

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- transformers >= 4.36.0
- torch >= 2.0.0
- peft >= 0.7.0
- unsloth >= 0.3.0
- deepspeed >= 0.12.0
- Additional utilities (numpy, scikit-learn, wandb, etc.)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── contrastive_nlp.py
└── ds_config.json
```

## Quick Start

1. **Setup Environment**:
   
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   
2. **Prepare Your Data**:
   ```python
   concepts = [
       "transistor",
       "capacitor",
       "resistor",
       # Add your EDA concepts
   ]
   
   descriptions = [
       "A semiconductor device used to amplify or switch electronic signals",
       "A passive electronic component that stores electrical energy in an electric field",
       "A passive electronic component that implements electrical resistance",
       # Add corresponding descriptions
   ]
   ```

3. **Train the Model**:
   ```bash
   deepspeed contrastive_nlp.py
   ```

## Configuration

### Model Configuration

The `ContrastiveLearningConfig` class in `contrastive_nlp.py` contains the main model parameters:

```python
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
```

### DeepSpeed Configuration

The `ds_config.json` file contains DeepSpeed-specific settings:
- ZeRO optimization (Stage 2)
- FP16 mixed precision training
- Gradient accumulation and clipping
- Learning rate scheduling

## Customization

### Negative Sample Generation

Customize the `create_negative_sample` function in `contrastive_nlp.py` for your specific EDA use case:

```python
def create_negative_sample(concept: str, description: str) -> str:
    """
    Implement your domain-specific negative sample generation strategy:
    1. Replace key components with similar but incorrect ones
    2. Modify circuit characteristics
    3. Change component relationships
    4. Alter technical specifications
    """
    pass
```

### Model Architecture

You can modify the model architecture by:
1. Changing the base model in `ContrastiveLearningConfig`
2. Adjusting the projection head dimensions
3. Modifying the LoRA configuration
4. Customizing the loss function

## Best Practices

1. **Data Preparation**:
   - Ensure concept-description pairs are accurate
   - Maintain consistent terminology
   - Include diverse EDA concepts
   - Balance the dataset

2. **Training**:
   - Start with default hyperparameters
   - Monitor training loss
   - Use validation set for early stopping
   - Save model checkpoints

3. **Negative Sampling**:
   - Implement domain-specific rules
   - Ensure negative samples are meaningful
   - Maintain technical accuracy
   - Consider component relationships

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests
3. Improving documentation
4. Adding more EDA-specific functionality

## License

MIT License

## Idea Of Paper

[Customized Retrieval Augmented Generation and Benchmarking for EDA Tool Documentation QA](https://arxiv.org/abs/2407.15353)
