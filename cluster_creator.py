import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loguru import logger
import pandas as pd

sys.path.append(".")
from CENSURED.common.common_utils import num_tokens_from_string
from CENSURED.common.secret_utils import get_config
from CENSURED.common.constants import SecretConstants
from CENSURED.azure_clients.openai_client import OpenAiClient
from CENSURED.azure_clients.aisearch_client import AISearchClient


N_ITER_KMEANS = 25 # can be adjusted
N_CLUSTERS = 37

PROMPT_GENERATE_CLUSTER_LABEL = (
            "Analiza el texto que te voy a proporcionar a continuación y genera tres o varias etiquetas que"
            " represente el contenido, con un máximo de 5 etiquetas."
            "Quiero que la etiqueta sea con una dos o tres palabras para dar detalle, y que clasifique el "
            "contenido por temas."
            "No incluyas introduccion ni texto adicional, solo saca las etiquetas que generas. "
            "Cuando saques las etiquetas, separalas por coma, no salto de linea ni otra cosa, para que cada etiqueta"
            " se separe de las demas con , Es decir con este estilo: etqueta1, etiqueta2, etiqueta3"
            "No incluyas la palabra 'Etiqueta' en la respuesta y asegúrate de que no esté en formato de lista."
            "Las etiquetas estan en lenguaje español."
        )


class ClusteringCreator:
    """
    Clase para crear clusters de embeddings. No se debería utilizar si no cambia la base de datos.
    """

    def __init__(self):
        # Configura AISearch
        search_service_url = get_config().get_value(SecretConstants.AISEARCH_ENDPOINT)
        logger.info("aisearch url: " + search_service_url)
        .......


def cluster_creator_main(df_clusters: pd.DataFrame, output_path: str="cluster_centroids_labels.csv"):
    """
    Ejecuta el clustering y guarda los resultados en un archivo CSV.

    :param pd.DataFrame df_clusters: DataFrame con los campos: document_id, embedding, document_path, content
    :param str output_path: Ruta del archivo de salida.
    """

    # df_clusters = pd.read_json("notebooks/documentos_pre.json")
    df_clusters = df_clusters[["document_id","content_vector", "document_path", "content"]].rename(columns={"content_vector":"embedding"})
    labeling_chunks_processor = ClusteringCreator()
    df_final = labeling_chunks_processor.execute_clustering(df_clusters, n_clusters=37, n_iter=10, max_iter=300, n_iter_cv=50)
    df_final.to_csv(
            output_path, index=False
        )  # queremos esto para hacer predicciones
