# sai_extractor.py

class TextExtractor:
    """Clase para extraer texto de estructuras de datos complejas."""

    @staticmethod
    def extract_text_recursive(content) -> str:
        """
        Extrae texto de forma recursiva de diferentes estructuras de datos.

        Args:
            content: Puede ser str, list, dict o cualquier otro tipo

        Returns:
            str: Texto extraído o cadena vacía si no hay contenido
        """
        # Caso 1: Ya es un string
        if isinstance(content, str):
            return content

        # Caso 2: Es una lista
        if isinstance(content, list):
            texts = []
            for item in content:
                # Recursión para cada elemento de la lista
                extracted = TextExtractor.extract_text_recursive(item)
                if extracted:
                    texts.append(extracted)
            return " ".join(texts)

        # Caso 3: Es un diccionario
        if isinstance(content, dict):
            # Prioridad 1: Si tiene "text", usarlo directamente
            if "text" in content:
                return str(content["text"])

            # Prioridad 2: Si tiene "content", recursión
            if "content" in content:
                return TextExtractor.extract_text_recursive(content["content"])

            # Prioridad 3: Si tiene "input_text" (formato Codex CLI)
            if "input_text" in content:
                return str(content["input_text"])

            # Prioridad 4: Si tiene "arguments" (formato Codex CLI)
            if "arguments" in content:
                return "EJECUTADO: " + str(content["arguments"])

            # Prioridad 5: Si tiene "output" (formato Codex CLI)
            if "output" in content:
                return "RESULTADO: " + str(content["output"])

            # Si no tiene ninguno de los campos esperados, retornar vacío
            return ""

        # Caso 4: Otro tipo (None, int, etc.)
        return ""


# Instancia global para facilitar el uso
text_extractor = TextExtractor()