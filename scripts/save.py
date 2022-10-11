from gcs import read_json
import requests
import fire
from tqdm import tqdm


def save(endpoint):
    records = read_json(file_name="embedded.json")
    headers = {
        "Accept": "application/sparql-results+json,*/*;q=0.9"
    }

    for record in tqdm(records, miniters=500):
        file_reference = record["thing"]
        try:
            q = f"""
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            
            DELETE {{
                GRAPH <http://mu.semte.ch/application> {{
                    <{file_reference}> ext:searchEmbedding ?oldEmbedding.
                }}
            }}
            WHERE {{
                GRAPH <http://mu.semte.ch/application> {{
                    OPTIONAL {{<{file_reference}> ext:searchEmbedding ?oldEmbedding.}}
                }}
            }}
            """

            r = requests.post(endpoint, data={"query": q}, headers=headers)
            if r.status_code != 200:
                print(f"[FAILURE] {50*'-'} /n {q} /n {50*'-'}")

            q = f"""
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            
            INSERT {{
                GRAPH <http://mu.semte.ch/application> {{
                    <{file_reference}> ext:searchEmbedding \"{list(record["embedding"])}\".
                    <{file_reference}> ext:ingestedByMl2GrowSmartRegulationsEmbedding "1".
                }}
            }}
            """
            r = requests.post(endpoint, data={"query": q}, headers=headers)

            if r.status_code != 200:
                print(f"[FAILURE] {50*'-'} /n {q} /n {50*'-'}")

        except Exception as ex:
            print(ex)
            raise ex


if __name__ == '__main__':
    fire.Fire(save)
