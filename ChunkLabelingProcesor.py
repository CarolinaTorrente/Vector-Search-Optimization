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
.....

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
