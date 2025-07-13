import re
import os
import pandas as pd
from typing import Optional, Set
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Set NLTK data path via environment variable
os.environ["NLTK_DATA"] = "/opt/render/nltk_data"

# Manual Indonesian stopwords sebagai fallback
INDONESIAN_STOPWORDS = {
    "ada",
    "adalah",
    "adanya",
    "adapun",
    "agak",
    "agaknya",
    "agar",
    "akan",
    "akankah",
    "akhir",
    "akhiri",
    "akhirnya",
    "aku",
    "akulah",
    "amat",
    "amatlah",
    "anda",
    "andalah",
    "antar",
    "antara",
    "antaranya",
    "apa",
    "apaan",
    "apabila",
    "apakah",
    "apalagi",
    "apatah",
    "artinya",
    "asal",
    "asalkan",
    "atas",
    "atau",
    "ataukah",
    "ataupun",
    "awal",
    "awalnya",
    "bagai",
    "bagaikan",
    "bagaimana",
    "bagaimanakah",
    "bagaimanapun",
    "bagi",
    "bagian",
    "bahkan",
    "bahwa",
    "bahwasanya",
    "baik",
    "bakal",
    "bakalan",
    "balik",
    "banyak",
    "bapak",
    "baru",
    "bawah",
    "beberapa",
    "begini",
    "beginian",
    "beginilah",
    "beginikah",
    "begitu",
    "begitukah",
    "begitulah",
    "begitupun",
    "bekerja",
    "belakang",
    "belakangan",
    "belum",
    "belumlah",
    "benar",
    "benarkah",
    "benarlah",
    "berada",
    "berakhir",
    "berakhirlah",
    "berakhirnya",
    "berapa",
    "berapakah",
    "berapalah",
    "berapapun",
    "berarti",
    "berawal",
    "berbagai",
    "berdatangan",
    "beri",
    "berikan",
    "berikut",
    "berikutnya",
    "berjumlah",
    "berkali",
    "berkata",
    "berkehendak",
    "berkeinginan",
    "berkenaan",
    "berlainan",
    "berlalu",
    "berlangsung",
    "berlebihan",
    "bermacam",
    "bermaksud",
    "bermula",
    "bersama",
    "bersedia",
    "bersiap",
    "bersoal",
    "bertanya",
    "bertanya-tanya",
    "berturut",
    "berturut-turut",
    "bertutur",
    "berujar",
    "berupa",
    "besar",
    "betul",
    "betulkah",
    "biasa",
    "biasanya",
    "bila",
    "bilakah",
    "bisa",
    "bisakah",
    "boleh",
    "bolehkah",
    "buat",
    "bukan",
    "bukankah",
    "bukanlah",
    "bukannya",
    "bulan",
    "bung",
    "cara",
    "caranya",
    "cukup",
    "cukupkah",
    "cukuplah",
    "cuma",
    "dahulu",
    "dalam",
    "dan",
    "dapat",
    "dari",
    "daripada",
    "datang",
    "dekat",
    "demi",
    "demikian",
    "demikianlah",
    "dengan",
    "depan",
    "di",
    "dia",
    "diakhiri",
    "diakhirinya",
    "dialah",
    "diantara",
    "diantaranya",
    "diberi",
    "diberikan",
    "diberikannya",
    "dibuat",
    "dibuatnya",
    "didapat",
    "didatangkan",
    "digunakan",
    "diibaratkan",
    "diibaratkannya",
    "diingat",
    "diingatkan",
    "diinginkan",
    "dijawab",
    "dijelaskan",
    "dijelaskannya",
    "dikarenakan",
    "dikatakan",
    "dikatakannya",
    "dikerjakan",
    "diketahui",
    "diketahuinya",
    "dikira",
    "dilakukan",
    "dilalui",
    "dilihat",
    "dimaksud",
    "dimaksudkan",
    "dimaksudkannya",
    "dimaksudnya",
    "diminta",
    "dimintai",
    "dimisalkan",
    "dimulai",
    "dimulailah",
    "dimulainya",
    "dimungkinkan",
    "dini",
    "dipastikan",
    "diperbuat",
    "diperbuatnya",
    "dipergunakan",
    "diperkirakan",
    "diperlihatkan",
    "diperlukan",
    "diperlukannya",
    "dipersoalkan",
    "dipertanyakan",
    "dipunyai",
    "diri",
    "dirinya",
    "disampaikan",
    "disebut",
    "disebutkan",
    "disebutkannya",
    "disini",
    "disinilah",
    "ditambahkan",
    "ditandaskan",
    "ditanya",
    "ditanyai",
    "ditanyakan",
    "ditegaskan",
    "ditujukan",
    "ditunjuk",
    "ditunjuki",
    "ditunjukkan",
    "ditunjukkannya",
    "ditunjuknya",
    "dituturkan",
    "dituturkannya",
    "diucapkan",
    "diucapkannya",
    "diungkapkan",
    "dong",
    "dulu",
    "empat",
    "enggak",
    "enggaknya",
    "entah",
    "entahlah",
    "guna",
    "gunakan",
    "hal",
    "hampir",
    "hanya",
    "hanyalah",
    "hari",
    "harus",
    "haruslah",
    "harusnya",
    "hendak",
    "hendaklah",
    "hendaknya",
    "hingga",
    "ia",
    "ialah",
    "ibarat",
    "ibaratkan",
    "ibaratnya",
    "ibu",
    "ikut",
    "ingat",
    "ingat-ingat",
    "ingin",
    "inginkah",
    "inginkan",
    "ini",
    "inikah",
    "inilah",
    "itu",
    "itukah",
    "itulah",
    "jadi",
    "jadilah",
    "jadinya",
    "jangan",
    "jangankan",
    "janganlah",
    "jauh",
    "jawab",
    "jawaban",
    "jawabnya",
    "jelas",
    "jelaskan",
    "jelaslah",
    "jelasnya",
    "jika",
    "jikalau",
    "juga",
    "jumlah",
    "jumlahnya",
    "justru",
    "kala",
    "kalau",
    "kalaulah",
    "kalaupun",
    "kalian",
    "kami",
    "kamilah",
    "kamu",
    "kamulah",
    "kan",
    "kapan",
    "kapankah",
    "kapanpun",
    "karena",
    "karenanya",
    "kasus",
    "kata",
    "katakan",
    "katakanlah",
    "katanya",
    "ke",
    "keadaan",
    "kebetulan",
    "kecil",
    "kedua",
    "keduanya",
    "keinginan",
    "kelamaan",
    "kelihatan",
    "kelihatannya",
    "keluar",
    "kemana",
    "kemari",
    "kemarin",
    "kemudian",
    "kenapa",
    "kepada",
    "kepadanya",
    "kesampaian",
    "keseluruhan",
    "keseluruhannya",
    "ketika",
    "ketika",
    "kini",
    "kinilah",
    "kira",
    "kira-kira",
    "kiranya",
    "kita",
    "kitalah",
    "kok",
    "kurang",
    "lagi",
    "lagian",
    "lah",
    "lain",
    "lainnya",
    "lalu",
    "lama",
    "lamanya",
    "lanjut",
    "lanjutnya",
    "lebih",
    "lewat",
    "lima",
    "luar",
    "macam",
    "maka",
    "makanya",
    "makin",
    "malah",
    "malahan",
    "mampu",
    "mampukah",
    "mana",
    "manakala",
    "manalagi",
    "masa",
    "masalah",
    "masalahnya",
    "masih",
    "masihkah",
    "masing",
    "masing-masing",
    "mau",
    "maupun",
    "melainkan",
    "melakukan",
    "melalui",
    "melihat",
    "melihatnya",
    "memang",
    "memastikan",
    "memberi",
    "memberikan",
    "membuat",
    "memerlukan",
    "memihak",
    "meminta",
    "memintakan",
    "memisalkan",
    "memperbuat",
    "mempergunakan",
    "memperkirakan",
    "memperlihatkan",
    "mempersiapkan",
    "mempersoalkan",
    "mempertanyakan",
    "mempunyai",
    "memulai",
    "memungkinkan",
    "menaiki",
    "menambah",
    "menambahkan",
    "menanti",
    "menantikan",
    "menanya",
    "menanyai",
    "menanyakan",
    "mendapat",
    "mendapatkan",
    "mendatang",
    "mendatangi",
    "mendatangkan",
    "menegaskan",
    "mengakhiri",
    "mengapa",
    "mengatakan",
    "mengatakannya",
    "mengenai",
    "mengerjakan",
    "mengetahui",
    "menggunakan",
    "menghendaki",
    "mengibaratkan",
    "mengibaratkannya",
    "mengingat",
    "mengingatkan",
    "menginginkan",
    "mengira",
    "mengucapkan",
    "mengucapkannya",
    "mengungkapkan",
    "menjadi",
    "menjawab",
    "menjelaskan",
    "menuju",
    "menunjuk",
    "menunjuki",
    "menunjukkan",
    "menunjuknya",
    "menurut",
    "menuturkan",
    "menyampaikan",
    "menyangkut",
    "menyatakan",
    "menyebutkan",
    "menyeluruh",
    "menyiapkan",
    "merasa",
    "mereka",
    "merekalah",
    "merupakan",
    "meski",
    "meskipun",
    "minta",
    "mirip",
    "misal",
    "misalkan",
    "misalnya",
    "mula",
    "mulai",
    "mulailah",
    "mulanya",
    "mungkin",
    "mungkinkah",
    "nah",
    "naik",
    "namun",
    "nanti",
    "nantinya",
    "nyaris",
    "oleh",
    "olehnya",
    "pada",
    "padahal",
    "padanya",
    "pak",
    "paling",
    "panjang",
    "pantas",
    "para",
    "pasti",
    "pastilah",
    "penting",
    "pentingnya",
    "per",
    "percuma",
    "perlu",
    "perlukah",
    "perlunya",
    "pernah",
    "pernahkah",
    "pertama",
    "pertama-tama",
    "pertanyaan",
    "pertanyakan",
    "pihak",
    "pihaknya",
    "pukul",
    "pula",
    "pun",
    "punya",
    "punyakah",
    "rah",
    "rasa",
    "rasanya",
    "rata",
    "rupanya",
    "saat",
    "saatnya",
    "saja",
    "sajalah",
    "saling",
    "sama",
    "sama-sama",
    "sambil",
    "sampai",
    "sampai-sampai",
    "sampaikan",
    "sana",
    "sangat",
    "sangatlah",
    "sani",
    "satu",
    "saya",
    "sayalah",
    "se",
    "sebab",
    "sebabnya",
    "sebagai",
    "sebagaimana",
    "sebagainya",
    "sebagian",
    "sebaik",
    "sebaik-baiknya",
    "sebaiknya",
    "sebaliknya",
    "sebanyak",
    "sebegini",
    "sebegitu",
    "sebelum",
    "sebelumnya",
    "sebenarnya",
    "seberapa",
    "sebesar",
    "sebetulnya",
    "sebisanya",
    "sebuah",
    "sebut",
    "sebutlah",
    "sebutnya",
    "secara",
    "secukupnya",
    "sedang",
    "sedangkan",
    "sedemikian",
    "sedikit",
    "sedikitnya",
    "seenaknya",
    "segala",
    "segalanya",
    "segera",
    "seharusnya",
    "sehingga",
    "seingat",
    "sejak",
    "sejauh",
    "sejenak",
    "sejumlah",
    "sekadar",
    "sekadarnya",
    "sekali",
    "sekali-kali",
    "sekalian",
    "sekaligus",
    "sekalipun",
    "sekarang",
    "sekarang",
    "sekecil",
    "seketika",
    "sekiranya",
    "sekitar",
    "sekitarnya",
    "sekurang-kurangnya",
    "sekurangnya",
    "sela",
    "selain",
    "selaku",
    "selama",
    "selama-lamanya",
    "selamanya",
    "selanjutnya",
    "seluruh",
    "seluruhnya",
    "semacam",
    "semakin",
    "semampu",
    "semampunya",
    "semasa",
    "semasih",
    "semata",
    "semata-mata",
    "semaunya",
    "sementara",
    "semisal",
    "semisalnya",
    "sempat",
    "semua",
    "semuanya",
    "semula",
    "sendiri",
    "sendirian",
    "sendirinya",
    "seolah",
    "seolah-olah",
    "seorang",
    "sepanjang",
    "sepantasnya",
    "sepantasnyalah",
    "seperlunya",
    "seperti",
    "sepertinya",
    "sepihak",
    "sering",
    "seringnya",
    "serta",
    "serupa",
    "sesaat",
    "sesama",
    "sesampai",
    "sesegera",
    "sesekali",
    "seseorang",
    "sesuatu",
    "sesuatunya",
    "sesudah",
    "sesudahnya",
    "setelah",
    "setempat",
    "setengah",
    "seterusnya",
    "setiap",
    "setiba",
    "setibanya",
    "setidak-tidaknya",
    "setidaknya",
    "setinggi",
    "seusai",
    "sewajarnya",
    "sewaktu",
    "siap",
    "siapa",
    "siapakah",
    "siapapun",
    "sih",
    "sini",
    "sinilah",
    "soal",
    "soalnya",
    "suatu",
    "sudah",
    "sudahkah",
    "sudahlah",
    "supaya",
    "tadi",
    "tadinya",
    "tahu",
    "tahun",
    "tak",
    "tambah",
    "tambahnya",
    "tampak",
    "tampaknya",
    "tandas",
    "tandasnya",
    "tanpa",
    "tanya",
    "tanyakan",
    "tanyanya",
    "tapi",
    "tegas",
    "tegasnya",
    "telah",
    "tempat",
    "tengah",
    "tentang",
    "tentu",
    "tentulah",
    "tentunya",
    "tepat",
    "terakhir",
    "terasa",
    "terbanyak",
    "terdahulu",
    "terdapat",
    "terdiri",
    "terhadap",
    "terhadapnya",
    "teringat",
    "teringat-ingat",
    "terjadi",
    "terjadilah",
    "terjadinya",
    "terkira",
    "terlalu",
    "terlebih",
    "terlihat",
    "termasuk",
    "ternyata",
    "tersampaikan",
    "tersebut",
    "tersebutlah",
    "tertentu",
    "tertuju",
    "terus",
    "terutama",
    "tetap",
    "tetapi",
    "tiap",
    "tiba",
    "tiba-tiba",
    "tidak",
    "tidakkah",
    "tidaklah",
    "tiga",
    "tinggi",
    "toh",
    "tunjuk",
    "turut",
    "tutur",
    "tuturnya",
    "ucap",
    "ucapnya",
    "ujar",
    "ujarnya",
    "umum",
    "umumnya",
    "ungkap",
    "ungkapnya",
    "untuk",
    "usah",
    "usai",
    "waduh",
    "wah",
    "wahai",
    "waktu",
    "waktunya",
    "walau",
    "walaupun",
    "wong",
    "yaitu",
    "yakin",
    "yakni",
    "yang",
}


def get_indonesian_stopwords() -> Set[str]:
    """
    Get Indonesian stopwords dengan fallback ke manual list

    Returns:
        Set[str]: Set berisi stopwords bahasa Indonesia
    """
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("indonesian"))
    except Exception as e:
        print(f"Using manual Indonesian stopwords (NLTK data not available): {e}")
        return INDONESIAN_STOPWORDS


def get_stemmer():
    """
    Cache stemmer untuk menghindari inisialisasi berulang

    Returns:
        Stemmer: Instance stemmer Sastrawi
    """
    if not hasattr(get_stemmer, "stemmer"):
        factory = StemmerFactory()
        get_stemmer.stemmer = factory.create_stemmer()
    return get_stemmer.stemmer


def get_cached_stopwords() -> Set[str]:
    """
    Cache stopwords untuk menghindari loading berulang

    Returns:
        Set[str]: Cached stopwords
    """
    if not hasattr(get_cached_stopwords, "stop_words"):
        get_cached_stopwords.stop_words = get_indonesian_stopwords()
    return get_cached_stopwords.stop_words


def preprocess_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
    """
    Preprocessing teks optimized untuk FastAPI dengan caching dan error handling

    Args:
        text (str): Teks yang akan diproses
        stop_words (Set[str], optional): Set stopwords custom. Defaults to None.

    Returns:
        str: Teks yang sudah diproses
    """

    # 1. Validasi input dan konversi ke string
    if not text or pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # 2. Konversi ke lowercase untuk normalisasi
    text = text.lower()

    # 3. Pembersihan karakter escape dan whitespace
    # Hapus tab, newline, dan karakter unicode escape
    text = re.sub(r"\\[tn]", " ", text)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
    text = re.sub(r"\\+", "", text)

    # 4. Pembersihan tanda baca dan karakter khusus
    # Hapus angka (pertimbangkan apakah angka penting untuk sistem rekomendasi)
    text = re.sub(r"\d+", "", text)

    # Hapus tanda baca kecuali huruf dan spasi
    text = re.sub(r"[^\w\s]", "", text)

    # 5. Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenisasi dan filtering
    tokens = [token for token in text.split() if token.strip() and len(token) > 1]

    if not tokens:
        return ""

    # 7. Stopword removal menggunakan cached stopwords
    if stop_words is None:
        stop_words = get_cached_stopwords()

    tokens = [word for word in tokens if word not in stop_words]

    if not tokens:
        return ""

    # 8. Stemming menggunakan cached stemmer
    try:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    except Exception as e:
        print(f"Stemming error: {e}")
        # Fallback: return tokens tanpa stemming
        pass

    # 9. Filter token kosong setelah stemming
    tokens = [token for token in tokens if token and len(token) > 1]

    return " ".join(tokens)


def preprocess_combined_features(
    deskripsi: str, kategori: str, kategori_weight: int = 1
) -> str:
    """
    Preprocessing khusus untuk combined features dengan weighted category

    Args:
        deskripsi (str): Deskripsi inovasi
        kategori (str): Kategori inovasi
        kategori_weight (int, optional): Bobot untuk kategori. Defaults to 1.

    Returns:
        str: Combined features yang sudah diproses
    """

    # Handle missing values
    deskripsi = str(deskripsi) if deskripsi and not pd.isna(deskripsi) else ""
    kategori = str(kategori) if kategori and not pd.isna(kategori) else ""

    # Gabungkan dengan weighted category
    if kategori_weight > 1:
        combined = deskripsi + " " + (kategori + " ") * kategori_weight
    else:
        combined = deskripsi + " " + kategori

    return preprocess_text(combined)


def batch_preprocess_text(
    texts: list[str], stop_words: Optional[Set[str]] = None
) -> list[str]:
    """
    Batch preprocessing untuk efisiensi ketika memproses banyak teks

    Args:
        texts (list[str]): List teks yang akan diproses
        stop_words (Set[str], optional): Set stopwords custom. Defaults to None.

    Returns:
        list[str]: List teks yang sudah diproses
    """

    # Cache stopwords dan stemmer sekali untuk semua teks
    if stop_words is None:
        stop_words = get_cached_stopwords()

    stemmer = get_stemmer()

    results = []
    for text in texts:
        try:
            processed = preprocess_text(text, stop_words)
            results.append(processed)
        except Exception as e:
            print(f"Error processing text: {e}")
            results.append("")

    return results


# Convenience function untuk backward compatibility
def process_innovation_text(deskripsi: str, kategori: str) -> str:
    """
    Wrapper function untuk memproses teks inovasi dengan format yang konsisten

    Args:
        deskripsi (str): Deskripsi inovasi
        kategori (str): Kategori inovasi

    Returns:
        str: Teks yang sudah diproses
    """
    return preprocess_combined_features(deskripsi, kategori)
