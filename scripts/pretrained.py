import logging
from abc import ABC

import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# from ts.metrics.dimension import Dimension
logger = logging.getLogger(__name__)

seed = 1
import random
import numpy as np

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logger = logging.getLogger(__name__)


# Returns grouped_entities=True
class TransformersClassifierHandler(ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers' checkpoint.
    """

    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.tokenizer = None
        self.model = None
        self._batch_size = 0
        self.initialized = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialize(self):

        model_path = "/models/RobertaModel_PDF_V1"
        # Read model serialize/pt file
        self.model = RobertaModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)

        # Turn off dropout
        self.model.eval()

        logger.debug(
            "Transformer model {0} loaded successfully".format(model_path)
        )

        self.initialized = True

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """

        # NOTE: This makes the assumption that your model expects text to be tokenized
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit
        # its expected input format.
        # print(type(inputs), inputs)
        def reprocess_encodings(x, max_length=512):
            resulting_input, input_idss, attention_masks = [], [], []

            for i in range(0, len(x), max_length):
                arr = x[i:i + max_length]

                if len(arr) != 512:
                    # Create zero padding tensor
                    input_ids, attention_mask = torch.zeros(512, dtype=torch.long), torch.zeros(512, dtype=torch.long)

                    # Update zero padding tensor with actual data
                    len_arr = len(arr)
                    input_ids[:len_arr] = torch.tensor(arr, dtype=torch.long)
                    attention_mask[:len_arr] = torch.ones(len_arr, dtype=torch.long)

                else:
                    input_ids = torch.tensor(arr, dtype=torch.long)
                    attention_mask = torch.ones(512, dtype=torch.long)

                input_idss.append(input_ids)
                attention_masks.append(attention_mask)

            # Batching them together
            input_ids_processed = torch.stack(input_idss)
            attention_ids_processed = torch.stack(attention_masks)

            return {"input_ids": input_ids_processed, "attention_mask": attention_ids_processed}

        # for input in inputs:
        texts = [" ".join(text.split()) for text in inputs]

        # I am aware that this is not actually fully batched, but that's not really relevant in this context.
        processed_embeddings = []
        for text in tqdm(texts, miniters=500):
            if len(text) != 0:
                with torch.no_grad():
                    # Get reprocessed input_ids and attention_masks
                    result = reprocess_encodings(
                        self.tokenizer.encode_plus(text, None, add_special_tokens=False)["input_ids"])

                    # Inference of the received input_ids/attention_mask
                    embedding = self.model(**result)["pooler_output"]

                    # Mean of resulting embeddings or simple squeeze to prep for next step
                    new_embedding = embedding.squeeze(0) if embedding.shape[0] == 1 else torch.mean(embedding, 0)

                    # Append embeddings (list) embeddings -
                    processed_embeddings.append(new_embedding.squeeze(0).tolist())
            else:
                processed_embeddings.append([])
                continue

        return {"texts": [{"embedding": item} for item in processed_embeddings]}
