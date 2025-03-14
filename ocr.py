CHAT_GPT_MAX_IMG_SIZE = 768

PROMPTS = [
    (
        "You are an advanced OCR model capable of not only reading text but also understanding and organizing it."
        " When you receive an image of a document, extract the text logically, considering text groupings, such as"
        " text boxes. If the document contains tables, return them in markdown format, accounting for merged cells."
        "If there are images within the text, describe their content precisely and explain any relation to other "
        "images or text. Focus on how images may contribute to understanding the document, such as diagrams or "
        "explanatory images. Provide all additional text in Spanish."
    ),
    (
        "You are the most advanced ORC model, you are capable of not only reading text but also understanding it and "
        "finding the text agrupations in a file, for example when there is a box of text in a document, you must "
        "cluster all the text inside that box. You are going to receive a image of a document and you must extract "
        "the text from it in a logical way. There can be tables in the document, return them in markdown format, "
        "keep in mind tables can have joined cells. If you see an image inside the text with text or something"
        " relevant, describe precisely its contents and if there is relation with the rest of the images as there can"
        " diagrams and the images can explain something, you have to explain what the images try to say. Answer only"
        " with the extracted text or images. All the text that is not originally in the document that you add "
        "must be in spanish."
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

        # Obtener las clases únicas
        unique_classes = np.unique(predictions_array)

        # Diccionario para almacenar los ranges de cada clase
        ranges = {c: [] for c in unique_classes}

        # Iterar sobre el array y sus índices
        for k, g in groupby(enumerate(predictions_array), key=lambda x: x[1]):
            group = list(g)
            start = group[0][0]
            end = group[-1][0]
            ranges[k].append(Range(start, end))
        print(ranges)

        return ranges

    @retry(retries=3, delay=1, backoff=2)
    def analyze_b64_with_document_intelligence(self, base64_encoded_pdf) -> list[str]:
        """
        Analiza un PDF en base64 con el servicio Document Intelligence

        :param base64_encoded_pdf: PDF en base64
        :return: Texto de cada página
        """
        analyze_request = {"base64Source": base64_encoded_pdf}
        poller = self.document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=analyze_request,
            output_content_format=ContentFormat.MARKDOWN,
        )
        result: AnalyzeResult = poller.result()
        spans = [page["spans"][0] for page in result.pages]
        pages_text = [
            result.content[span["offset"] : span["offset"] + span["length"]]
            for span in spans
        ]
        return pages_text

    def process_ranges_with_document_intelligence(
        self, pdf_path: str, ranges: list[Range]
    ) -> list[TextRange]:
        """
        Procesa las imágenes con el modelo Document Intelligence y devuelve el texto de cada rango

        :param di_client: Document Intelligence client
        :param pdf_path: Ruta al documento PDFaml 
        :param ranges: Lista de rangos a procesar
        :return: Lista de con el texto obtenido con Document Intelligence para cada rango
        """
        n_pages = sum(r.end - r.start + 1 for r in ranges)
        logger.info(f"Processing {n_pages} pages with Document Intelligence")
        processed_ranges = []
        for r in ranges:
            pdf_range_b64 = get_pdf_range_b64(pdf_path, r)
            text = self.analyze_b64_with_document_intelligence(pdf_range_b64)
            processed_ranges.append(TextRange(r, text))

        return processed_ranges

    def process_ranges_with_chatgpt(
        self,
        pdf_path,
        pages: list[np.ndarray],
        ranges: list[Range],
    ) -> list[TextRange]:
        """
        Process the images with the ChatGPT model and return the text of each range

        :param openai_client: OpenAiClient instance
        :param pages: list of images
        :param ranges: list of ranges to process
        :param document_intelligence_client: DocIntelligence CLient
        :param pdf_path: string url del pdf a procesar

        :return: list of TextRange instances with the text obtained with ChatGPT for each range
        """
        processed_ranges = []
        for rang in ranges:
            text = self.process_range_with_chatgpt(pdf_path, pages, rang)
            processed_ranges.append(TextRange(rang, text))

        return processed_ranges

    def process_range_with_chatgpt(
        self,
        pdf_path,
        pages: list[np.ndarray],
        rang: Range,
    ) -> list[str]:
        """
        Procesa un rango de páginas con el modelo ChatGPT y devuelve el texto de cada página

        :param OpenAiClient openai_client: OpenAiClient instance
        :param DocumentIntelligenceClient document_intelligence_client: DocumentIntelligenceClient instance
        :param str pdf_path: Ruta al documento PDF
        :param list[np.ndarray] pages: Lista de imágenes
        :param Range rang: Rango de páginas a procesar
        :return: Lista de textos de cada página
        """
        imgs_to_process = pages[rang.start : rang.end + 1]
        pages_text = []
        for idx, img in enumerate(imgs_to_process):
            # Reducimos tamaño y color y devolvemos base 64
            if image_utils.is_empty_image(img):
                response = ""
            else:
                try:
                    response = self.openai_client.prompt_image(
                        self.gpt_model_name,
                        img,
                        get_prompt(),
                    )
                    if response.strip() == "":
                        response = self._fallback_to_document_intelligence(
                            pdf_path,
                            Range(rang.start + idx, rang.start + idx),
                        )
                except Exception as ex:
                    logger.info(
                        f"Error processing page {idx} with ChatGPT, fallback to document intelligence: {ex}"
                    )
                    response = self._fallback_to_document_intelligence(
                        pdf_path,
                        Range(rang.start + idx, rang.start + idx),
                    )

            pages_text.append(response)

        return pages_text

    def _fallback_to_document_intelligence(
        self,
        pdf_path,
        rang: Range,
    ) -> list[str]:
        textrange = self.process_ranges_with_document_intelligence(
            pdf_path,
            [Range(rang.start, rang.start)],
        )
        response = textrange[0].text[0]
        return response

    @staticmethod
    def _add_page_metadata(text, page_number) -> str:
        """
        Añade el número de página al texto

        :param str text: Texto al que añadir el número de página
        :param int page_number: Número de página
        :return str: Texto con el número de página añadido
        """

        return f"<! -- iniciostartinicio pagina {page_number} -- >\n{text}\n<! -- finendfin pagina {page_number} -- >"

    @staticmethod
    def _join_ranges(pages_di, pages_chatgpt):
        """
        Une los rangos de Document Intelligence y ChatGPT

        :param list[TextRange] pages_di: Lista de rangos de Document Intelligence
        :param list[TextRange] pages_chatgpt: Lista de rangos de ChatGPT
        :return: Lista de rangos con el texto obtenido con Document Intelligence y ChatGPT
        """
        all_pages = pages_di + pages_chatgpt
        if len(all_pages) == 0:
            return ""

        all_pages.sort(key=lambda x: x.range.start)

        all_pages = [page for range_text in all_pages for page in range_text.text]

        all_pages = [
            OCR._add_page_metadata(text, page) for page, text in enumerate(all_pages, 1)
        ]

        text = "\n".join([text for text in all_pages])
        text = parse_utils.replace_multiple_hashes(text)
        return text

    def ocr_image(self, img: np.ndarray) -> str:
        """
        OCR an image with the OpenAI API or Document Intelligence and return the response

        :param img: numpy array representing the image
        :param openai_client: OpenAiClient instance
        :param document_intelligence_client: DocumentIntelligenceClient instance
        :return: response from the OpenAI API
        """
        try:
            prediction = self.aml_client.predict_all([img])[0]
            if prediction == "document_intelligence":
                result = " ".join(
                    self.analyze_b64_with_document_intelligence(
                        image_utils.img_to_b64(img)
                    )
                ).strip()
                return result
            else:
                return self.openai_client.prompt_image(
                    model=get_gpt_model_name(),
                    img=img,
                    system_prompt=get_prompt(),
                )
        except Exception as ex:
            logger.error(f"Error OCRing image: {ex}")
            return "".join(
                self.analyze_b64_with_document_intelligence(image_utils.img_to_b64(img))
            ).strip()

    # @line_profiler.profile
    def process_pdf_to_mark(self, pdf_path):
        """
        Process a PDF document by sending it to Document Intelligence, computing AML predictions,
        and replacing specific pages with ChatGPT output where needed, optimized for speed.

        :param str pdf_path: Path to the PDF file.
        :return: Extracted text in Markdown format.
        """
        logger.info(f"Processing PDF: {pdf_path}")
        try:
            # 1-preparar para Document Intelligence submission
            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

            #2-PDF to images
            logger.info("Converting PDF to images")
            pages = list(
                image_utils.pdf_to_numpy_images_generator(
                    pdf_path, min_size=CHAT_GPT_MAX_IMG_SIZE))

            #3- AML predictions, ChatGPT, and Document Intelligence concurrently
            logger.info("Submitting tasks for AML predictions, ChatGPT processing, and Document Intelligence")
            with ThreadPoolExecutor(max_workers=3) as executor:
                aml_future = executor.submit(self.aml_client.predict_all, pages)
                docint_future = executor.submit(
                    self.analyze_b64_with_document_intelligence, base64_pdf)

                # Placeholder for ChatGPT processing (no ranges available yet)
                chatgpt_future = None

                # Wait for AML predictions
                predictions = aml_future.result()

                # Step 4: Process ranges requiring ChatGPT in parallel
                ranges = self._get_prediction_ranges(predictions)
                chatgpt_ranges = ranges.get("chatgpt", [])
                if chatgpt_ranges:
                    logger.info(f"Processing {len(chatgpt_ranges)} ranges with ChatGPT")
                    chatgpt_future = executor.submit(self.process_ranges_with_chatgpt, pdf_path, pages, chatgpt_ranges)

                # Gather ChatGPT results if applicable
                chatgpt_results = chatgpt_future.result() if chatgpt_future else []
                
                #esperar o/p de docint para juntar resultado
                docint_text = docint_future.result()

            # Step 5: Merge Document Intelligence and ChatGPT results
            logger.info("Merging Document Intelligence and ChatGPT results")
            final_pages = []
            for i, page_text in enumerate(docint_text):
                # Check if this page is replaced by ChatGPT output
                replaced = False
                for chatgpt_range in chatgpt_results:
                    if chatgpt_range.range.start <= i <= chatgpt_range.range.end:
                        chatgpt_page_index = i - chatgpt_range.range.start
                        final_pages.append(chatgpt_range.text[chatgpt_page_index])
                        replaced = True
                        break
                if not replaced:
                    final_pages.append(page_text)

            # Combine the pages into final markdown text
            final_text = "\n".join(
                self._add_page_metadata(page_text, idx + 1)
                for idx, page_text in enumerate(final_pages)
            )
            logger.info("Final combined text prepared")

            return final_text

        except Exception as ex:
            logger.error(f"Error processing PDF: {pdf_path} - {ex}")
            raise Exception(
                f"Error processing PDF: {pdf_path} - {ex}\n"
                f"Traceback: {traceback.format_exc()}"
            ) from ex
