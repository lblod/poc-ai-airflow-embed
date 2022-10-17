import fire

from gcs import read_json, write_json
from pretrained import TransformersClassifierHandler


def embed(*args):
    try:
        model = TransformersClassifierHandler()
        records = read_json(file_name="export.json")

        if not model.initialized:
            model.initialize()

        if records is None:
            return None
        text = [t["text"][:10_000] for t in records]

        # Call model inference on text
        embeddings = model.inference(text)

        # Re-order the embeddings in order to write to embedded.json for next step
        embeddings = [{**records[i], "embedding": embedding["embedding"]} for i, embedding in
                      enumerate(embeddings["texts"])]
        write_json(file_name="embedded.json", content=embeddings)

    except Exception as e:
        print(e)
        raise e


if __name__ == '__main__':
    fire.Fire(embed)
