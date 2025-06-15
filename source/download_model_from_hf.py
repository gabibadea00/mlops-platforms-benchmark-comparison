from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

def download_huggingface_model(model_name: str, save_path: str):
    """
    Downloads a Hugging Face model and tokenizer to a specified path.

    Args:
        model_name (str): The model ID from Hugging Face (e.g., "bert-base-uncased", "CAMeL-Lab/bert-base-arabic-camelbert-mix", "dkleczek/modern_bert").
        save_path (str): The local directory where the model and tokenizer should be saved.

    Example usage: download_huggingface_model("dkleczek/modern_bert", "./models/modern_bert")
    """
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading model '{model_name}' to '{save_path}'...")
    
    # Download config, tokenizer, and model
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Save everything locally
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"Model '{model_name}' downloaded and saved to '{save_path}'")

def download_quantized_model(model_name: str, save_path: str, quant_bits: int = 8):
    """
    Downloads a quantized version of a Hugging Face model using bitsandbytes.
    This function could be useful when comparing models that we train with other models that are
    already on HuggingFace

    Args:
        model_name (str): The model ID from Hugging Face (must support quantization).
        save_path (str): The directory to save the model files.
        quant_bits (int): Quantization bits (either 8 or 4).
    """
    from transformers import BitsAndBytesConfig

    os.makedirs(save_path, exist_ok=True)

    quant_config = BitsAndBytesConfig(
        load_in_8bit=(quant_bits == 8),
        load_in_4bit=(quant_bits == 4),
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True,
    )

    print(f"Downloading {quant_bits}-bit quantized model '{model_name}' to '{save_path}'...")

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)

    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"Quantized model saved to '{save_path}'")