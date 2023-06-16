# OCR Model Training Using Vision Transformer

This project involves the development of an Optical Character Recognition (OCR) model using the Vision Transformer (ViT) architecture from the Hugging Face Transformers library. The OCR task involves converting images of handwritten or printed text into machine-encoded text. The model is fine-tuned on OCR datasets - RVL-CDIP, IIIT5K, and SVT, allowing it to read and recognize text from input images efficiently.

## Data Preparation 

The first step involves downloading and preparing the RVL-CDIP, IIIT5K, and SVT datasets. We:

- Resize the images to a uniform size 
- Normalize their pixel values 
- Split the datasets into training and validation sets
- Convert them into PyTorch datasets
- Define a data loader for each to facilitate batch loading

## Model Training

The training process utilizes the VisionEncoderDecoderModel from the Hugging Face Transformers library. We:

- Define special tokens for creating decoder_input_ids from labels
- Ensure the vocab size is set correctly
- Set beam search parameters like eos_token_id, max_length, early_stopping, no_repeat_ngram_size, and length_penalty
- Define an optimizer for the model
- Train the model over a specified number of epochs
- Perform loss computation and weight updates during training

## Model Evaluation 

Following training, we evaluate the model's performance using the validation dataset. We:

- Calculate the Character Error Rate (CER) for the model's predictions
- Compare it with the ground truth labels
- Generate model predictions using the validation dataset
- Calculate the CER for these predictions, printing the validation CER metric after each epoch

## Saving the Model 

Once the model is trained and evaluated, we save it and the tokenizer to a directory using the `save_pretrained` method from the Hugging Face Transformers library. This enables us to reuse the trained model and tokenizer for OCR predictions on new images.

In summary, this project results in an OCR model that leverages the Vision Transformer architecture and is fine-tuned on several OCR datasets. The model is evaluated and saved for future OCR predictions on new images.
