import random
import tiktoken
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from loguru import logger
from CENSURED.common.secret_utils import get_config
from CENSURED.search.documents import SearchClient
from CENSURED.core.credentials import AzureKeyCredential
from CENSURED.common.constants import SecretConstants, AISearchConstants
from CENSURED.azure_clients.openai_client import OpenAiClient
from CENSURED.azure_clients.aisearch_client import AISearchClient


class LabelingChunksProcessor:

    def __init__(self):
        # Configura AISearch
        search_service_url = get_config().get_value(SecretConstants.AISEARCH_ENDPOINT)
        logger.info("aisearch url: " + search_service_url)
        search_api_key = get_config().get_value(SecretConstants.AISEARCH_KEY)
        self.index_name = get_config().get_value(SecretConstants.DOCUMENTS_INDEX_NAME)

        self.model = get_config().get_value(SecretConstants.OPENAI_MODEL_MINI)
        self.assistant = "You are a expert in labeling content to make clusters more easy to select"
        logger.info("model: " + self.model)
        azure_endpoint = get_config().get_value(SecretConstants.OPENAI_URL)
        logger.info("openai url: " + azure_endpoint)
        self.api_version = "2024-02-01"
        self.client = OpenAiClient(api_version=self.api_version)

        # Crea el cliente de búsqueda, indice AISearch de cada entorno
        credential = AzureKeyCredential(search_api_key)
        # self.search_client = SearchClient(endpoint=search_service_url, index_name=self.index_name, credential=credential)
        self.search_client = AISearchClient(self.index_name)


    # np.random.seed(42)

    @staticmethod
    def contar_tokens(texto, modelo="gpt-3.5-turbo"):
        # Cargar el codificador específico para el modelo
        codificador = tiktoken.encoding_for_model(modelo)
        # Convertir el texto en tokens
        tokens = codificador.encode(texto)
        # Contar la cantidad de tokens
        cantidad_tokens = len(tokens)
        return cantidad_tokens


    # Obtener todos los documentos de la base de datos
    def get_all_docs(self):
        results = self.search_client.search(search_text="*", top=None, include_total_count=True)
        documents = [result for result in results]
        return documents

    # Obtener todos los embeddings sin agrupar por document_id
    def get_all_embeddings(self):
        results = self.get_all_docs()
        logger.info(f"Embeddings tratados: {len(results)}")
        embeddings = []
        document_ids = []  # Para mantener una referencia a los IDs de los documentos
        document_paths = []  # Para mantener los document_path
        contents = []  # Para mantener el contenido de cada documento
        total_documents = len(results)

        for doc in results:
            document_id = doc.get('id')  # ID del documento (nuevo campo)
            document_path = doc.get('document_path')  # Path del documento
            embedding = doc.get('content_vector')  # Vector de embedding
            content = doc.get('content')  # Contenido del documento

            if embedding is not None:
                embeddings.append(embedding)  # Añadir el embedding directamente a la lista
                document_ids.append(document_id)  # Mantener un registro del document_id
                document_paths.append(document_path)  # Mantener un registro del document_path
                contents.append(content)  # Almacenar el contenido también

        logger.info(f'Número total de documentos en el estudio de cluster: {total_documents} documentos')
        logger.info(f'Número total de embeddings procesados: {len(embeddings)}')

        return np.array(embeddings), document_ids, document_paths, contents  # Devolver también el contenido


    # Realizar clustering usando K-Means
    @staticmethod
    def cluster_kmeans(embeddings, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        inertia = kmeans.inertia_  # Inercia
        silhouette_avg = silhouette_score(embeddings, cluster_labels)  # Silhouette Score
        return cluster_labels, inertia, silhouette_avg, kmeans.cluster_centers_


    # Función para generar etiquetas llamando al modelo de GPT
    def generate_cluster_label(self, contents):
        prompt = (
            "Analiza el texto que te voy a proporcionar a continuación y genera tres o varias etiquetas que"
            " represente el contenido, con un máximo de 5 etiquetas."
            "Quiero que la etiqueta sea con una dos o tres palabras para dar detalle, y que clasifique el "
            "contenido por temas."
            "No incluyas introduccion ni texto adicional, solo saca las etiquetas que generas. "
            "Cuando saques las etiquetas, separalas por coma, no salto de linea ni otra cosa, para que cada etiqueta"
            " se separe de las demas con , Es decir con este estilo: etqueta1, etiqueta2, etiqueta3"
            "No incluyas la palabra 'Etiqueta' en la respuesta y asegúrate de que no esté en formato de lista."
            "Etiquetas estan en lenguaje español.\n\n\n"
        )
        combined_contents = "\n".join(contents)  # Añadir los contenidos
        total_tokens = self.contar_tokens(prompt) + self.contar_tokens(combined_contents)

        if total_tokens > 125000:
            # Si excede el límite, truncar o dividir los contenidos
            combined_contents = " ".join(contents)[:125000 - self.contar_tokens(prompt)]  # Truncar para ajustarse
            contents = [combined_contents]  # Usar el contenido truncado

        prompt += combined_contents  # Añadir el contenido truncado

        # Llamar a la función call_chatgpt_model para obtener la etiqueta
        etiqueta = self.client.call_chatgpt_model(prompt, self.model, self.assistant)
        return etiqueta

    def execute_clustering(self):
        logger.info("Comenzando el clustering")
        # Procesar los embeddings
        embeddings, document_ids, document_paths, contents = self.get_all_embeddings()  # Obtener también contenidos
        # Rango de k a probar
        #max_range = len(self.get_all_docs())
        #max_range = int(math.sqrt(max_range) / 2)
        #k_values = range(10, max_range)
        #inertia_values = []
        #silhouette_scores = []

        ''' # Probar K-Means para cada valor de k
        for k in k_values:
            cluster_labels, inertia, silhouette_avg, _ = cluster_kmeans(embeddings, k)
            inertia_values.append(inertia)
            silhouette_scores.append(silhouette_avg)

            # Imprimir los valores de inercia y silhouette por cada k
            logger.info(f'k: {k}, Inercia: {inertia}, Silhouette Score: {silhouette_avg}')'''

        '''# Imprimir el mejor valor de k basado en el Silhouette Score
        best_k = k_values[np.argmax(silhouette_scores)]'''

        # Realizar el clustering final usando el mejor k
        final_cluster_labels, _, _, cluster_centers = LabelingChunksProcessor.cluster_kmeans(embeddings, 37)

        # Crear un diccionario para contar el número de embeddings en cada clúster
        clusters_count = {}
        clusters_dict = {}

        for label in final_cluster_labels:
            if label not in clusters_count:
                clusters_count[label] = 0
                clusters_dict[label] = []  # Inicializa la lista para este cluster
            clusters_count[label] += 1

        # Diccionario para almacenar las etiquetas de los clústeres
        cluster_labels = {}
        document_labels = []  # Lista para almacenar la info del document_id, document_path, content_vector y etiqueta

        for label, count in clusters_count.items():
            logger.info(f'Clúster {label}: {count} embeddings')
            # Obtener los embeddings y el contenido en el clúster actual
            cluster_embeddings = embeddings[final_cluster_labels == label]
            cluster_ids = np.array(document_ids)[final_cluster_labels == label]  # Obtener document_id
            cluster_paths = np.array(document_paths)[final_cluster_labels == label]  # Obtener document_path
            cluster_texts = np.array(contents)[final_cluster_labels == label]  # Obtener contenidos

            # Calcular la distancia de cada embedding al centroide del clúster
            distances = cdist([cluster_centers[label]], cluster_embeddings, metric='euclidean')[0]

            # Crear un diccionario para almacenar los documentos por clúster
            clusters_dict[label] = []
            for doc_id, embedding, labels, path in zip(document_ids, embeddings, final_cluster_labels, document_paths):
                if labels not in clusters_dict:
                    clusters_dict[labels] = []
                clusters_dict[labels].append((doc_id, embedding, path))

            # Calcular el centroide del clúster
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calcular la distancia de cada embedding al centroide
            distances = cdist([centroid], cluster_embeddings, metric='euclidean')[0]

            # Ordenar los documentos por la distancia al centroide
            sorted_docs = sorted(zip(cluster_ids, distances), key=lambda x: x[1])

            # Seleccionar los 100 documentos más cercanos al centroide
            closest_docs = [(doc_id, path) for (doc_id, _), path in zip(sorted_docs[:100], cluster_paths)]

            # Seleccionar los 100 documentos más lejanos al centroide
            farthest_docs = [(doc_id, path) for (doc_id, _), path in zip(sorted_docs[-100:], cluster_paths)]

            # Obtener el contenido de los documentos más cercanos y más lejanos al centroide
            # Obtener el contenido de los documentos más cercanos y más lejanos al centroide
            closest_contents = [doc_id for doc_id, _ in closest_docs] + [doc_id for doc_id, _ in farthest_docs]

            # Interleave closest and farthest documents
            mixed_contents = []
            min_length = min(len(closest_docs), len(farthest_docs))

            for i in range(min_length):
                mixed_contents.append(closest_docs[i])
                mixed_contents.append(farthest_docs[i])

            # If there are leftover documents in either list, add them
            mixed_contents.extend(closest_docs[min_length:])
            mixed_contents.extend(farthest_docs[min_length:])

            # Extract document paths or contents for labeling
            mixed_contents_texts = [path for _, path in mixed_contents]  # Extracting paths or actual contents

            # Generate a label for the cluster using GPT
            etiqueta = self.generate_cluster_label(mixed_contents_texts)

            logger.info(f'Etiqueta generada para el clúster {label}: {etiqueta} ')

            # Almacenar la etiqueta en el diccionario de clústeres
            cluster_labels[label] = etiqueta

            # Asignar etiqueta a cada documento en el clúster
            for doc_id, doc_path, vector, content in zip(cluster_ids, cluster_paths, cluster_embeddings, cluster_texts):
                document_labels.append({
                    "id": doc_id,
                    "document_path": doc_path,
                    "content_vector": vector,
                    "etiqueta": etiqueta
                })

        # Actualizar el campo "etiqueta_cluster" en cada documento
        results = self.search_client.search(search_text="*", include_total_count=True)
        documentos_para_actualizar = []

        # Recorrer todos los documentos y modificar el campo "etiqueta_cluster" con su ID
        for doc in results:
            documento_id = doc["id"]  # Asumiendo que el campo 'id' es el identificador único

            # Buscar la etiqueta correspondiente usando el ID del documento
            # Aquí se asume que hay una relación entre `document_labels` y `documento_id`
            etiqueta_correspondiente = next((d['etiqueta'] for d in document_labels if d['id'] == documento_id), None)

            if etiqueta_correspondiente is not None:
                # Crear un nuevo documento que solo contiene el ID y el campo que quieres modificar
                documento_modificado = {
                    "id": documento_id,  # El campo 'id' es necesario
                    "etiqueta_cluster": etiqueta_correspondiente  # Asignar el nuevo valor a 'etiqueta_cluster'
                }

                # Añadir el documento modificado a la lista de documentos para actualizar
                documentos_para_actualizar.append(documento_modificado)

        # Verificar si hay documentos para actualizar
        if documentos_para_actualizar:
            # Usar la función merge_documents para actualizar todos los documentos de la lista
            self.search_client.merge_documents(documentos_para_actualizar)
            logger.info(f"Documentos actualizados: {len(documentos_para_actualizar)}")
        else:
            logger.info("No se encontraron documentos para actualizar.")
