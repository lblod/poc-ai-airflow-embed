import fire
import requests
from tqdm import tqdm

from gcs import read_json


def save(endpoint):
    """
    This save function will write the created query to the given endpoint. This query is specific for each service.

    :param endpoint: the url where the sparql endpoint is hosted on
    :return:
    """
    records = read_json(file_name="embedded.json")
    headers = {
        "Accept": "application/sparql-results+json,*/*;q=0.9"
    }

    # iterate over all the records in a file
    for record in tqdm(records, miniters=500):
        file_reference = record["thing"]
        try:
            q = f"""
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            
             DELETE{{
             GRAPH <http://mu.semte.ch/application>{{
             <{file_reference}> ext:searchEmbedding ?embed; ext:ingestedByMl2GrowSmartRegulationsEmbedding ?sre . 
            }}
            }}
            WHERE{{
             <{file_reference}> ext:searchEmbedding ?embed; ext:ingestedByMl2GrowSmartRegulationsEmbedding ?sre .            
            }}
            """

            # request for the sparql DELETE WHERE statement above
            r = requests.post(endpoint, data={"query": q}, headers=headers)
            if r.status_code != 200:
                print(f"[FAILURE] {50 * '-'} /n {q} /n {50 * '-'}")

            q = f"""
            PREFIX ext: <http://mu.semte.ch/vocabularies/ext/>
            
            INSERT {{
                GRAPH <http://mu.semte.ch/application> {{
                    <{file_reference}> ext:searchEmbedding \"{list(record["embedding"])}\".
                    <{file_reference}> ext:ingestedByMl2GrowSmartRegulationsEmbedding "1".
                }}
            }}
            """

            # request for the sparql INSERT
            r = requests.post(endpoint, data={"query": q}, headers=headers)

            if r.status_code != 200:
                print(f"[FAILURE] {50 * '-'} /n {q} /n {50 * '-'}")

        # basic exception handeling, easy to read in airflow logs
        except Exception as ex:

            print(ex)
            raise ex


if __name__ == '__main__':
    fire.Fire(save)
