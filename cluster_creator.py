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
        self.index_name = get_config().get_value(SecretConstants.DOCUMENTS_INDEX_NAME)

        self.model = get_config().get_value(SecretConstants.OPENAI_MODEL_MINI)
        self.assistant = (
            "You are a expert in labeling content to make clusters more easy to select"
        )
        logger.info("model: " + self.model)
        azure_endpoint = get_config().get_value(SecretConstants.OPENAI_URL)
        logger.info("openai url: " + azure_endpoint)
        self.api_version = "2024-02-01"
        self.client = OpenAiClient(api_version=self.api_version)

        # Crea el cliente de búsqueda, indice AISearch de cada entorno
        # self.search_client = SearchClient(endpoint=search_service_url, index_name=self.index_name, credential=credential)
        self.search_client = AISearchClient(self.index_name)

    # np.random.seed(42)

    @staticmethod
    def count_tokens(texto) -> int:
        """
        Cuenta los tokens de un texto.
        :param str texto: El texto a contar.
        :return int: La cantidad de tokens.
        """
        return num_tokens_from_string(texto)

    # Obtener todos los documentos de la base de datos
    def get_all_docs(self):
        results = self.search_client.search(
            search_text="*", top=None, include_total_count=True
        )
        documents = [result for result in results]
        return documents

    # Obtener todos los embeddings sin agrupar por document_id
    def get_all_embeddings(self) -> tuple[list[list[float]], list[str], list[str], list[str]]:
        """
        Obtiene todos los embeddings sin agrupar por document_id
        :return tuple: Tupla con los embeddings, document_ids, document_paths y contents
        """
        results = self.get_all_docs()
        logger.info(f"Embeddings tratados: {len(results)}")
        embeddings = []
        document_ids = []  # Para mantener una referencia a los IDs de los documentos
        document_paths = []  # Para mantener los document_path
        contents = []  # Para mantener el contenido de cada documento
        total_documents = len(results)

        for doc in results:
            document_id = doc.get("id")  # ID del documento (nuevo campo)
            document_path = doc.get("document_path")  # Path del documento
            embedding = doc.get("content_vector")  # Vector de embedding
            content = doc.get("content")  # Contenido del documento

            if embedding is not None:
                embeddings.append(
                    embedding
                )  # Añadir el embedding directamente a la lista
                document_ids.append(document_id)  # Mantener un registro del document_id
                document_paths.append(
                    document_path
                )  # Mantener un registro del document_path
                contents.append(content)  # Almacenar el contenido también

        logger.info(
            f"Número total de documentos en el estudio de cluster: {total_documents} documentos"
        )
        logger.info(f"Número total de embeddings procesados: {len(embeddings)}")

        return (
            np.array(embeddings),
            document_ids,
            document_paths,
            contents,
        )  # Devolver también el contenido

    # Realizar clustering usando K-Means
    @staticmethod
    def cluster_kmeans(embeddings, k, n_iter=10, max_iter=300, n_iter_cv=5) -> tuple[list[int], float, float, list[list[float]]]:
        """
        Realiza clustering usando K-Means.

        :param list[list[float]] embeddings: Lista de embeddings a clasificar.
        :param int k: Número de clusters a crear.
        :param int n_iter: Número de iteraciones para el clustering.
        :param int max_iter: Número máximo de iteraciones para el clustering.
        :param int n_iter_cv: Número de iteraciones para la validación cruzada.
        :return tuple: Tupla con las etiquetas, la inercia, el mejor silhouette score y los centroides.
        """
        mejor_silhouette = -1
        mejor_kmeans = None
        mejor_etiquetas = None
        mejor_inertia = None

        for _ in range(n_iter_cv):  # Realizamos 5 iteraciones
            kmeans = KMeans(n_clusters=k, n_init=n_iter, max_iter=max_iter)
            etiquetas = kmeans.fit_predict(embeddings)
            inertia = kmeans.inertia_
            silhouette_avg = silhouette_score(embeddings, etiquetas)

            if silhouette_avg > mejor_silhouette:
                mejor_silhouette = silhouette_avg
                mejor_kmeans = kmeans
                mejor_etiquetas = etiquetas
                mejor_inertia = inertia

        logger.info(f"Mejor Silhouette Score: {mejor_silhouette}")

        return (
            mejor_etiquetas,
            mejor_inertia,
            mejor_silhouette,
            mejor_kmeans.cluster_centers_,
        )

    def generate_cluster_label(self, contents):
        """
        Genera etiquetas para un grupo de contenidos usando GPT.

        :param list[str] contents: Lista de contenidos a etiquetar.
        :return str: Etiqueta generada.
        """
        prompt = PROMPT_GENERATE_CLUSTER_LABEL
        combined_contents = "\n".join(contents)  # Añadir los contenidos
        total_tokens = self.count_tokens(prompt) + self.count_tokens(
            combined_contents
        )

        if total_tokens > 125000:
            # Si excede el límite, truncar o dividir los contenidos
            combined_contents = " ".join(contents)[
                : 125000 - self.count_tokens(prompt)
            ]  # Truncar para ajustarse
            contents = [combined_contents]  # Usar el contenido truncado

        prompt += combined_contents  # Añadir el contenido truncado

        # Llamar a la función call_chatgpt_model para obtener la etiqueta
        etiqueta = self.client.call_chatgpt_model(prompt, self.model, self.assistant)
        return etiqueta


    def get_all_docs_df(self) -> pd.DataFrame:
        """
        Obtiene todos los documentos y los convierte en un DataFrame
        :return pd.DataFrame: DataFrame con los campos: document_id, embedding, document_path, content
        """
        embeddings, document_ids, document_paths, contents = (
            self.get_all_embeddings()
        )  # Obtener también contenidos
        df_clusters = pd.DataFrame(
            zip(document_ids, embeddings, document_paths, contents),
            columns=[
                "document_id",
                "embedding",
                "document_path",
                "content",
            ],
        )
        return df_clusters

    def execute_clustering(
        self,
        df_clusters: pd.DataFrame = None,
        n_clusters: int = N_CLUSTERS,
        n_iter: int = N_ITER_KMEANS,
        max_iter: int = 300,
        n_iter_cv: int = 5,
    ) -> pd.DataFrame:
        """
        Este método ejecuta el clustering y genera las etiquetas para los clústeres.
        Recibe un Dataframe con los campos: document_id, embedding, document_path, content
        Si no se proporciona, se obtienen todos los documentos de la base de datos

        :param pd.DataFrame df_clusters: DataFrame con los campos: document_id, embedding, document_path, content
        :param int n_clusters: Número de clusters a crear
        :param int n_iter: Número de iteraciones para el clustering
        :param int max_iter: Número máximo de iteraciones para el clustering
        :param int n_iter_cv: Número de iteraciones para la validación cruzada
        :return pd.DataFrame: DataFrame con los campos: cluster_label, centroid, etiqueta
        """

        logger.info("Comenzando el clustering")
        # Procesar los embeddings

        if df_clusters is None:
            logger.info(
                "No se proporcionó un DataFrame de clusters, obteniendo todos los documentos"
            )
            df_clusters = self.get_all_docs_df()

        # Normalizar los embeddings
        df_clusters["embedding"] = df_clusters["embedding"].apply(lambda x: x / np.linalg.norm(x))

        # Realizar el clustering final usando el mejor k
        final_cluster_labels, _, _, cluster_centers = (
            ClusteringCreator.cluster_kmeans(
                df_clusters["embedding"].tolist(), n_clusters, n_iter, max_iter, n_iter_cv
            )
        )

        centroids_dict = {i: vec.tolist() for i, vec in enumerate(cluster_centers)}

        df_clusters["cluster_label"] = final_cluster_labels

        # Diccionario para almacenar las etiquetas de los clústeres

        df_clusters["centroid"] = df_clusters["cluster_label"].map(centroids_dict)
        df_sample = df_clusters.groupby("cluster_label").sample(frac=0.3)

        def generar_etiqueta(grupo: pd.DataFrame):
            mixed_contents_texts = grupo["content"].tolist()
            etiqueta = self.generate_cluster_label(mixed_contents_texts)
            logger.info(f"Etiqueta generada para el clúster {grupo.name}: {etiqueta}")
            return etiqueta

        df_grouped = df_sample.groupby("cluster_label").apply(generar_etiqueta)
        df_final = df_sample[["cluster_label", "centroid"]].drop_duplicates(
            subset=["cluster_label"]
        )
        df_final["etiqueta"] = df_final["cluster_label"].map(df_grouped)
        df_final["etiqueta"] = df_final["etiqueta"].str.capitalize()

        return df_final


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
