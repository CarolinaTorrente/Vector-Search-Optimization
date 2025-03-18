CHAT_GPT_MAX_IMG_SIZE = 768

PROMPTS = [
    (
        "You are an advanced OCR model capable of not only reading text but also understanding and org..."
    ),
    (
        "You are the most advanced ORC model, you are capable of not only reading text but also ..."
    ),
]


get_prompt = partial(choice, PROMPTS)


def get_gpt_model_name():
    """
    Obtiene el nombre del modelo configurado

    :return str: nombre del modelo
    """
    model_name = get_config().get_value(SecretConstants.OPENAI_MODEL)
    if is_empty_string(model_name):
        model_name = "gpt-4o"

    return model_name


class OCR:
    """
    Clase para procesar documentos con Document Intelligence y ChatGPT.
    """

    def __init__(
        self,
        document_intelligence_client: DocumentIntelligenceClient,
        openai_client: OpenAiClient,
    ):
        self.document_intelligence_client: DocumentIntelligenceClient = (
            document_intelligence_client
        )
        self.openai_client: OpenAiClient = openai_client
        self.gpt_model_name: str = get_gpt_model_name()
        self.aml_client: AMLClient = AMLClient()

    @staticmethod
    def _get_prediction_ranges(predictions_array: np.array) -> dict[str, list[Range]]:
        """
        Get the ranges of each class in the predictions array
        For example in an array like ["a", "a", "b", "b", "b", "a", "a", "b"]
        The output would be {"a": [Range(0, 1), Range(5, 6)], "b": [Range(2, 4), Range(7, 7)]}

        :param predictions_array: array with the predictions

        :return: dictionary with the ranges of each class
        """
        # Convertir a array de NumPy si no lo es ya
        predictions_array = np.array(predictions_array)

  .....
