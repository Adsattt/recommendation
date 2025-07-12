import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

nltk.data.path.append("/opt/render/nltk_data")


def preprocess_text(text: str, stop_words=None) -> str:
    """
    Preprocessing teks dengan stemming dan menghapus stopwords
    Args:
        text (str): Teks yang akan diproses
        stop_words (set, optional): Set stopwords. Defaults to None.
    Returns:
        str: Teks yang sudah diproses
    """
    if stop_words is None:
        stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Lowercase
    text = text.lower()

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)

    tokens = [word for word in text.split() if word not in stop_words]

    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)
