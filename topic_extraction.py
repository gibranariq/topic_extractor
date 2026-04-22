# pip install pandas emoji yake pydantic nlp-id
from __future__ import annotations

import json
import re
import string
import sys
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import Counter
from time import perf_counter

import emoji
import pandas as pd
import yake
from nlp_id.postag import PosTag

from configs.config import settings


# POS tagger sekali inisialisasi
PART_OF_SPEECH_TAGGER = PosTag()


# ============================ Text Utils ============================

def strip_emoji(text: str) -> str:
    """Hapus semua emoji (safe untuk None)."""
    return emoji.replace_emoji(text or "", replace="")

def normalize_spaces(text: str, keep_newlines: bool) -> str:
    """
    - keep_newlines=True  : rapikan spasi tapi JAGA newline (buat POS sentence split).
    - keep_newlines=False : flatten semua whitespace jadi satu spasi (buat YAKE).
    """
    if keep_newlines:
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
        return t.strip()
    else:
        return re.sub(r"\s+", " ", text).strip()


def lowercase_first_paragraph_keep_rest(text: str) -> str:
    """
    Turunkan paragraf pertama (judul), sisanya biarkan.
    Gandakan paragraf pertama 3x buat boosting.
    Jika tidak ada blank line, turunkan baris pertama saja dan gandakan 3x.
    """
    if not isinstance(text, str):
        return ""

    # normalisasi newline
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", t)

    if not paras:
        return ""

    # kasus hanya 1 paragraf (tanpa blank line)
    if len(paras) == 1:
        lines = paras[0].split("\n")
        if not lines:
            return ""
        lines[0] = lines[0].lower()
        boosted_first = "\n".join([lines[0]] * 3)
        return boosted_first + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")

    # ada beberapa paragraf
    paras[0] = paras[0].lower()
    boosted_title = "\n".join([paras[0]] * 3)
    return "\n\n".join([boosted_title] + paras[1:])

def clean_preserve_case(text: str) -> str:
    """
    Cleaner untuk POS:
    - hapus emoji
    - rapikan spasi
    - JAGA newline & kapital (biar POS bagus)
    """
    if not isinstance(text, str):
        return ""
    t = strip_emoji(text)
    return normalize_spaces(t, keep_newlines=True)

def to_yake_source_from_pos(pos_ready_text: str) -> str:
    """
    Final text buat YAKE:
    - pakai hasil clean_preserve_case (emoji sudah dibuang)
    - flatten whitespace
    - lowercase
    """
    return normalize_spaces(pos_ready_text, keep_newlines=False).lower()


def normalize_for_match(phrase: str) -> str:
    """Normalisasi buat equality/subsumption (trim, strip punct, casefold)."""
    collapsed = re.sub(r"\s+", " ", phrase).strip()
    stripped = collapsed.strip(string.punctuation + "“”‘’\"")
    return stripped.casefold()

def toks(phrase: str) -> List[str]:
    """Tokenize sederhana by whitespace (untuk post-processing)."""
    return [t for t in re.split(r"\s+", phrase.strip()) if t]


# ============================ Stopwords ============================

def load_stopwords(path_str: str | None) -> set[str]:
    """Load stopwords .txt (satu entri per baris), disimpan lowercase."""
    if not path_str:
        return set()
    path = Path(path_str)
    if not path.exists():
        print(f"[WARN] Stopwords file not found: {path}", file=sys.stderr)
        return set()
    entries = [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sw = set(entries)
    print(f"[INFO] Loaded {len(sw)} custom stopwords from {path}")
    return sw


# ============================ POS Helpers ============================

def compute_pos_map_with_title_lowercased(raw_text: str) -> Dict[str, List[str]]:
    """
    Kurangi false NNP dari Title Case: turunkan judul, bersihkan (tanpa hilang newline),
    lalu POS tag. Return: token_lower -> list of raw tags (bisa multi).
    """
    t = lowercase_first_paragraph_keep_rest(raw_text)
    t = clean_preserve_case(t)
    pairs = PART_OF_SPEECH_TAGGER.get_pos_tag(t)

    pos_map: Dict[str, List[str]] = {}
    for token, tag in pairs:
        token = (token or "").strip()
        if not token:
            continue
        pos_map.setdefault(token.lower(), []).append(tag)
    return pos_map

def most_frequent_tag(tags: List[str]) -> str:
    """Ambil tag dominan untuk menstabilkan keputusan per token."""
    if not tags:
        return "UNK"
    return Counter(tags).most_common(1)[0][0]

def build_pos_filtered_text(original_text: str, allowed_tags: set[str]) -> str:
    """
    Filter teks berdasarkan POS (pre-extraction):
    - Simpan token yang tag-nya ∈ allowed_tags
    - Biarkan tanda baca ringan agar tetap natural
    - Hasil akhirnya masih preserve case; YAKE akan lower di step berikutnya
    """
    pairs = PART_OF_SPEECH_TAGGER.get_pos_tag(original_text)
    kept: List[str] = []
    for token, tag in pairs:
        token = (token or "").strip()
        if not token:
            continue
        if token in {".", ",", ":", ";", "!", "?"}:
            kept.append(token)
        elif tag in allowed_tags:
            kept.append(token)
    text_keep = " ".join(kept)
    text_keep = re.sub(r"\s+([.,:;!?])", r"\1", text_keep)
    return normalize_spaces(text_keep, keep_newlines=False)


# ============================ YAKE ============================

def build_yake_extractor(stopwords_iterable: Iterable[str] | None) -> yake.KeywordExtractor:
    return yake.KeywordExtractor(
        lan=settings.yake_language,
        n=settings.yake_max_ngram,
        top=settings.yake_initial_top_k,  # set besar; kita postprocess dulu baru slice final 10
        dedupLim=settings.yake_deduplication_limit,
        dedupFunc=settings.yake_deduplication_function,
        windowsSize=settings.yake_window_size,
        features=None,
        stopwords=stopwords_iterable,
    )

def yake_all_pairs_sorted(text_for_yake: str, extractor: yake.KeywordExtractor) -> List[Tuple[str, float]]:
    """
    Jalankan YAKE dan kembalikan SEMUA kandidat (sesuai 'top' dari extractor),
    disortir ascending by score (skor kecil = lebih baik).
    """
    if not text_for_yake:
        return []
    pairs = extractor.extract_keywords(text_for_yake)
    return sorted(pairs, key=lambda x: x[1])


# ============================ Dedup Rules (A & B) ============================

def postprocess_rule_ab(
    keyword_score_pairs: List[Tuple[str, float]],
    final_top_k: int,
    drop_all_stopwords_phrases: bool,
    custom_stopwords_iterable: Iterable[str] | None,
) -> List[Tuple[str, float]]:
    """
    Rule A: jika unigram A dan unigram B ada, serta bigram "A B" ada → ambil bigram, drop A & B.
    Rule B: uniqueness by token-overlap: frasa yang share token dengan yang sudah di-keep di-skip.
    """
    sw_norm = {normalize_for_match(sw) for sw in (custom_stopwords_iterable or [])}

    # 1) prefilter frasa yang semua token-nya stopword
    pre: List[Tuple[str, float]] = []
    for phrase, score in keyword_score_pairs:
        p = phrase.strip()
        if drop_all_stopwords_phrases and sw_norm:
            tkn = [normalize_for_match(t) for t in toks(p)]
            if tkn and all(t in sw_norm for t in tkn):
                continue
        pre.append((p, score))
    if not pre:
        return []

    # 2) normalisasi token per frasa
    norm_tokens: Dict[str, List[str]] = {p: [normalize_for_match(t) for t in toks(p)] for p, _ in pre}

    # 3) deteksi bigram "terlindungi" (gabungan langsung dari dua unigram yang juga ada)
    unigram_phrases = {p for p, _ in pre if len(norm_tokens[p]) == 1}
    unigram_token_to_phrase = {norm_tokens[p][0]: p for p in unigram_phrases}

    protected_bigrams: set[str] = set()
    for p, _ in pre:
        ts = norm_tokens[p]
        if len(ts) == 2:
            a, b = ts
            if a in unigram_token_to_phrase and b in unigram_token_to_phrase:
                protected_bigrams.add(p)

    # 4) urutkan: bigram terlindungi dulu, lalu skor YAKE
    pre.sort(key=lambda item: (0 if item[0] in protected_bigrams else 1, item[1]))

    # 5) greedy select dengan blocking token (Rule B) + blokir unigram komponen bigram
    kept: List[Tuple[str, float]] = []
    used_tokens: set[str] = set()
    block_unigrams: set[str] = set()

    for p, s in pre:
        ts = set(norm_tokens[p])

        if p in protected_bigrams:
            if used_tokens.isdisjoint(ts):
                kept.append((p, s))
                used_tokens.update(ts)
                a, b = norm_tokens[p]
                if a in unigram_token_to_phrase:
                    block_unigrams.add(unigram_token_to_phrase[a])
                if b in unigram_token_to_phrase:
                    block_unigrams.add(unigram_token_to_phrase[b])
            continue

        if p in block_unigrams:
            continue

        if used_tokens.isdisjoint(ts):
            kept.append((p, s))
            used_tokens.update(ts)

        if len(kept) >= final_top_k:
            break

    return kept


# ============================ System Info (optional) ============================

def print_system_info() -> None:
    print("\n" + "=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    try:
        import os
        print(f"CPU cores: {os.cpu_count()}")
    except Exception:
        pass
    print("=" * 60 + "\n")


# ============================ Pipeline ============================

def run_pipeline() -> None:
    print_system_info()

    # 1) load data
    df = pd.read_csv(settings.input_csv)
    if settings.text_column not in df.columns:
        raise KeyError(f"Column `{settings.text_column}` not found in CSV. Available: {list(df.columns)}")
    print(f"[INFO] Loaded {len(df)} rows from {settings.input_csv}")

    # 2) load stopwords & build YAKE
    custom_stopwords = load_stopwords(settings.stopwords_path)
    yake_extractor = build_yake_extractor(custom_stopwords or None)

    # 3) loop dokumen
    keywords_final: List[List[str]] = []
    keyword_pairs_final: List[List[Tuple[str, float]]] = []
    keyword_pairs_all: List[List[Tuple[str, float]]] = []
    keywords_with_pos: List[List[dict]] = []
    time_ms_list: List[float] = []

    for raw_text in df[settings.text_column].astype(str):
        t0 = perf_counter()

        # a) text untuk POS: turunkan judul (plus boosting 3x) → bersihin (emoji dibuang, newline dipertahankan)
        text_for_pos = lowercase_first_paragraph_keep_rest(raw_text)
        text_for_pos = clean_preserve_case(text_for_pos)

        # b) POS map (audit; stabil pakai most_frequent_tag)
        try:
            pos_map = compute_pos_map_with_title_lowercased(raw_text)
        except Exception as err:
            print(f"[WARN] POS map error, empty map used: {err}", file=sys.stderr)
            pos_map = {}

        # c) pre-extraction POS filter (opsional → default True di config kamu)
        source_text = (
            build_pos_filtered_text(text_for_pos, settings.allowed_part_of_speech_tags)
            if settings.enable_pre_extraction_pos_filter
            else text_for_pos
        )

        # d) teks YAKE (emoji sudah bersih; flatten whitespace; lowercase)
        yake_text = to_yake_source_from_pos(source_text)

        # e) YAKE — ambil SEMUA kandidat (sesuai 'top' di extractor), sort by score
        initial_pairs_sorted = yake_all_pairs_sorted(yake_text, yake_extractor)
        keyword_pairs_all.append(initial_pairs_sorted)  # audit semua kandidat (sorted)

        # f) APPLY RULES DULU (A/B) di FULL SET, TANPA slice dulu
        #    kasih final_top_k besar supaya rules milih tanpa dipotong
        kept_pairs_full = postprocess_rule_ab(
            initial_pairs_sorted,
            final_top_k=10**9,                   # biarin rules jalan dulu sepenuhnya
            drop_all_stopwords_phrases=True,
            custom_stopwords_iterable=custom_stopwords,
        )

        # g) BARU SLICE 10 TERATAS (FINAL)
        kept_pairs = kept_pairs_full[:settings.yake_final_top_k]

        # h) timing
        time_ms_list.append((perf_counter() - t0) * 1000.0)

        # i) persist + POS audit
        final_phrases = [p for p, _ in kept_pairs]
        keywords_final.append(final_phrases)
        keyword_pairs_final.append(kept_pairs)

        pos_audit = [
            {"phrase": p, "pos": [most_frequent_tag(pos_map.get(tok.lower(), [])) for tok in toks(p)]}
            for p in final_phrases
        ]
        keywords_with_pos.append(pos_audit)

    # 4) tulis kolom hasil
    df["keywords"] = keywords_final
    df["keywords_str"] = df["keywords"].apply(lambda xs: ", ".join(xs) if xs else "")
    if settings.keep_scores_column:
        df["keywords_scored"] = [json.dumps(pairs, ensure_ascii=False) for pairs in keyword_pairs_final]
    df["keywords_scored_all"] = [json.dumps(pairs, ensure_ascii=False) for pairs in keyword_pairs_all]
    df["keywords_with_pos"] = [json.dumps(records, ensure_ascii=False) for records in keywords_with_pos]
    df["time_to_10_keywords_ms_total"] = time_ms_list

    # 5) save
    Path(settings.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.output_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved → {settings.output_csv}")
    print(df[["keywords_str", "keywords_with_pos", "keywords_scored_all"]].head(3))

    # ringkas timing
    try:
        print(
            "[TIMING total] mean = {:.1f} ms | median = {:.1f} ms | max = {:.1f} ms | min = {:.1f} ms".format(
                df["time_to_10_keywords_ms_total"].mean(),
                df["time_to_10_keywords_ms_total"].median(),
                df["time_to_10_keywords_ms_total"].max(),
                df["time_to_10_keywords_ms_total"].min(),
            )
        )
    except Exception:
        pass


if __name__ == "__main__":
    run_pipeline()


# # models/topic_extraction.py
# # pip install pandas emoji yake pydantic nlp-id
# from __future__ import annotations

# import json
# import re
# import string
# import sys
# import platform
# from pathlib import Path
# from typing import Dict, Iterable, List, Tuple
# from collections import Counter
# from time import perf_counter

# import emoji
# import pandas as pd
# import yake
# from nlp_id.postag import PosTag

# from configs.config import settings


# # POS tagger sekali inisialisasi
# PART_OF_SPEECH_TAGGER = PosTag()


# # ============================ Text Utils ============================

# def strip_emoji(text: str) -> str:
#     """Hapus semua emoji (safe untuk None)."""
#     return emoji.replace_emoji(text or "", replace="")

# def normalize_spaces(text: str, keep_newlines: bool) -> str:
#     """
#     - keep_newlines=True  : rapikan spasi tapi JAGA newline (buat POS sentence split).
#     - keep_newlines=False : flatten semua whitespace jadi satu spasi (buat YAKE).
#     """
#     if keep_newlines:
#         t = text.replace("\r\n", "\n").replace("\r", "\n")
#         t = re.sub(r"[ \t]+", " ", t)
#         t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
#         return t.strip()
#     else:
#         return re.sub(r"\s+", " ", text).strip()

# # def lowercase_first_paragraph_keep_rest(text: str) -> str:
# #     """
# #     Turunkan paragraf pertama (judul), sisanya biarkan.
# #     Paragraf dipisah blank line; kalau nggak ada, turunkan baris pertama saja.
# #     """
# #     if not isinstance(text, str):
# #         return ""
# #     t = text.replace("\r\n", "\n").replace("\r", "\n")
# #     paras = re.split(r"\n\s*\n", t)
# #     if not paras:
# #         return ""
# #     if len(paras) == 1:
# #         lines = paras[0].split("\n")
# #         if not lines:
# #             return ""
# #         lines[0] = lines[0].lower()
# #         return "\n".join(lines)
# #     paras[0] = paras[0].lower()
# #     return "\n\n".join(paras)

# def lowercase_first_paragraph_keep_rest(text: str) -> str:
#     """
#     Turunkan paragraf pertama (judul), sisanya biarkan.
#     Gandakan paragraf pertama 3x buat boosting.
#     Jika tidak ada blank line, turunkan baris pertama saja dan gandakan 3x.
#     """
#     if not isinstance(text, str):
#         return ""

#     # normalisasi newline
#     t = text.replace("\r\n", "\n").replace("\r", "\n")
#     paras = re.split(r"\n\s*\n", t)

#     if not paras:
#         return ""

#     # kasus hanya 1 paragraf (tidak ada blank line)
#     if len(paras) == 1:
#         lines = paras[0].split("\n")
#         if not lines:
#             return ""
#         # turunkan baris pertama
#         lines[0] = lines[0].lower()
#         # gandakan baris pertama 3x untuk efek boost
#         boosted_first = "\n".join([lines[0]] * 3)
#         # gabungkan baris pertama (boosted) + sisa baris
#         return boosted_first + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")

#     # kalau punya lebih dari satu paragraf (ada blank line)
#     paras[0] = paras[0].lower()
#     boosted_title = "\n".join([paras[0]] * 3)
#     return "\n\n".join([boosted_title] + paras[1:])

# def clean_preserve_case(text: str) -> str:
#     """
#     Cleaner untuk POS:
#     - hapus emoji
#     - rapikan spasi
#     - JAGA newline & kapital (biar POS bagus)
#     """
#     if not isinstance(text, str):
#         return ""
#     t = strip_emoji(text)
#     return normalize_spaces(t, keep_newlines=True)

# def to_yake_source_from_pos(pos_ready_text: str) -> str:
#     """
#     Final text buat YAKE:
#     - pakai hasil clean_preserve_case (emoji sudah dibuang)
#     - flatten whitespace
#     - lowercase
#     """
#     return normalize_spaces(pos_ready_text, keep_newlines=False).lower()


# def normalize_for_match(phrase: str) -> str:
#     """Normalisasi buat equality/subsumption (trim, strip punct, casefold)."""
#     collapsed = re.sub(r"\s+", " ", phrase).strip()
#     stripped = collapsed.strip(string.punctuation + "“”‘’\"")
#     return stripped.casefold()

# def toks(phrase: str) -> List[str]:
#     """Tokenize sederhana by whitespace (untuk post-processing)."""
#     return [t for t in re.split(r"\s+", phrase.strip()) if t]


# # ============================ Stopwords ============================

# def load_stopwords(path_str: str | None) -> set[str]:
#     """Load stopwords .txt (satu entri per baris), disimpan lowercase."""
#     if not path_str:
#         return set()
#     path = Path(path_str)
#     if not path.exists():
#         print(f"[WARN] Stopwords file not found: {path}", file=sys.stderr)
#         return set()
#     entries = [
#         line.strip().lower()
#         for line in path.read_text(encoding="utf-8").splitlines()
#         if line.strip()
#     ]
#     sw = set(entries)
#     print(f"[INFO] Loaded {len(sw)} custom stopwords from {path}")
#     return sw


# # ============================ POS Helpers ============================

# def compute_pos_map_with_title_lowercased(raw_text: str) -> Dict[str, List[str]]:
#     """
#     Kurangi false NNP dari Title Case: turunkan judul, bersihkan (tanpa hilang newline),
#     lalu POS tag. Return: token_lower -> list of raw tags (bisa multi).
#     """
#     t = lowercase_first_paragraph_keep_rest(raw_text)
#     t = clean_preserve_case(t)
#     pairs = PART_OF_SPEECH_TAGGER.get_pos_tag(t)

#     pos_map: Dict[str, List[str]] = {}
#     for token, tag in pairs:
#         token = (token or "").strip()
#         if not token:
#             continue
#         pos_map.setdefault(token.lower(), []).append(tag)
#     return pos_map

# def most_frequent_tag(tags: List[str]) -> str:
#     """Ambil tag dominan untuk menstabilkan keputusan per token."""
#     if not tags:
#         return "UNK"
#     return Counter(tags).most_common(1)[0][0]

# def build_pos_filtered_text(original_text: str, allowed_tags: set[str]) -> str:
#     """
#     Filter teks berdasarkan POS (pre-extraction):
#     - Simpan token yang tag-nya ∈ allowed_tags
#     - Biarkan tanda baca ringan agar tetap natural
#     - Hasil akhirnya masih preserve case; YAKE akan lower di step berikutnya
#     """
#     pairs = PART_OF_SPEECH_TAGGER.get_pos_tag(original_text)
#     kept: List[str] = []
#     for token, tag in pairs:
#         token = (token or "").strip()
#         if not token:
#             continue
#         if token in {".", ",", ":", ";", "!", "?"}:
#             kept.append(token)
#         elif tag in allowed_tags:
#             kept.append(token)
#     text_keep = " ".join(kept)
#     text_keep = re.sub(r"\s+([.,:;!?])", r"\1", text_keep)
#     return normalize_spaces(text_keep, keep_newlines=False)


# # ============================ YAKE ============================

# def build_yake_extractor(stopwords_iterable: Iterable[str] | None) -> yake.KeywordExtractor:
#     return yake.KeywordExtractor(
#         lan=settings.yake_language,
#         n=settings.yake_max_ngram,
#         top=settings.yake_initial_top_k,
#         dedupLim=settings.yake_deduplication_limit,
#         dedupFunc=settings.yake_deduplication_function,
#         windowsSize=settings.yake_window_size,
#         features=None,
#         stopwords=stopwords_iterable,
#     )

# def yake_topk_pairs(text_for_yake: str, extractor: yake.KeywordExtractor, k: int) -> List[Tuple[str, float]]:
#     """Jalankan YAKE dan ambil top-k (keyword, score) dengan skor kecil = lebih baik."""
#     if not text_for_yake:
#         return []
#     pairs = extractor.extract_keywords(text_for_yake)
#     return sorted(pairs, key=lambda x: x[1])[:k]


# # ============================ Dedup Rules (A & B) ============================

# def postprocess_rule_ab(
#     keyword_score_pairs: List[Tuple[str, float]],
#     final_top_k: int,
#     drop_all_stopwords_phrases: bool,
#     custom_stopwords_iterable: Iterable[str] | None,
# ) -> List[Tuple[str, float]]:
#     """
#     Rule A: jika unigram A dan unigram B ada, serta bigram "A B" ada → ambil bigram, drop A & B.
#     Rule B: uniqueness by token-overlap: frasa yang share token dengan yang sudah di-keep di-skip.
#     """
#     sw_norm = {normalize_for_match(sw) for sw in (custom_stopwords_iterable or [])}

#     # 1) prefilter frasa yang semua token-nya stopword
#     pre: List[Tuple[str, float]] = []
#     for phrase, score in keyword_score_pairs:
#         p = phrase.strip()
#         if drop_all_stopwords_phrases and sw_norm:
#             tkn = [normalize_for_match(t) for t in toks(p)]
#             if tkn and all(t in sw_norm for t in tkn):
#                 continue
#         pre.append((p, score))
#     if not pre:
#         return []

#     # 2) normalisasi token per frasa
#     norm_tokens: Dict[str, List[str]] = {p: [normalize_for_match(t) for t in toks(p)] for p, _ in pre}

#     # 3) deteksi bigram "terlindungi" (gabungan langsung dari dua unigram yang juga ada)
#     unigram_phrases = {p for p, _ in pre if len(norm_tokens[p]) == 1}
#     unigram_token_to_phrase = {norm_tokens[p][0]: p for p in unigram_phrases}

#     protected_bigrams: set[str] = set()
#     for p, _ in pre:
#         ts = norm_tokens[p]
#         if len(ts) == 2:
#             a, b = ts
#             if a in unigram_token_to_phrase and b in unigram_token_to_phrase:
#                 protected_bigrams.add(p)

#     # 4) urutkan: bigram terlindungi dulu, lalu skor YAKE
#     pre.sort(key=lambda item: (0 if item[0] in protected_bigrams else 1, item[1]))

#     # 5) greedy select dengan blocking token (Rule B) + blokir unigram komponen bigram
#     kept: List[Tuple[str, float]] = []
#     used_tokens: set[str] = set()
#     block_unigrams: set[str] = set()

#     for p, s in pre:
#         ts = set(norm_tokens[p])

#         if p in protected_bigrams:
#             if used_tokens.isdisjoint(ts):
#                 kept.append((p, s))
#                 used_tokens.update(ts)
#                 a, b = norm_tokens[p]
#                 if a in unigram_token_to_phrase:
#                     block_unigrams.add(unigram_token_to_phrase[a])
#                 if b in unigram_token_to_phrase:
#                     block_unigrams.add(unigram_token_to_phrase[b])
#             continue

#         if p in block_unigrams:
#             continue

#         if used_tokens.isdisjoint(ts):
#             kept.append((p, s))
#             used_tokens.update(ts)

#         if len(kept) >= final_top_k:
#             break

#     return kept


# # ============================ System Info (optional) ============================

# def print_system_info() -> None:
#     print("\n" + "=" * 60)
#     print("SYSTEM INFO")
#     print("=" * 60)
#     print(f"OS: {platform.system()} {platform.release()}")
#     print(f"Platform: {platform.platform()}")
#     print(f"Processor: {platform.processor()}")
#     print(f"Machine: {platform.machine()}")
#     print(f"Python: {platform.python_version()}")
#     try:
#         import os
#         print(f"CPU cores: {os.cpu_count()}")
#     except Exception:
#         pass
#     print("=" * 60 + "\n")


# # ============================ Pipeline ============================

# def run_pipeline() -> None:
#     print_system_info()

#     # 1) load data
#     df = pd.read_csv(settings.input_csv)
#     if settings.text_column not in df.columns:
#         raise KeyError(f"Column `{settings.text_column}` not found in CSV. Available: {list(df.columns)}")
#     print(f"[INFO] Loaded {len(df)} rows from {settings.input_csv}")

#     # 2) load stopwords & build YAKE
#     custom_stopwords = load_stopwords(settings.stopwords_path)
#     yake_extractor = build_yake_extractor(custom_stopwords or None)

#     # 3) loop dokumen
#     keywords_final: List[List[str]] = []
#     keyword_pairs_final: List[List[Tuple[str, float]]] = []
#     keyword_pairs_all: List[List[Tuple[str, float]]] = []
#     keywords_with_pos: List[List[dict]] = []
#     time_ms_list: List[float] = []

#     for raw_text in df[settings.text_column].astype(str):
#         t0 = perf_counter()

#         # a) text untuk POS: turunkan judul → bersihin (emoji sudah dibuang, newline dipertahankan)
#         text_for_pos = lowercase_first_paragraph_keep_rest(raw_text)
#         text_for_pos = clean_preserve_case(text_for_pos)

#         # b) POS map (audit; stabil pakai most_frequent_tag)
#         try:
#             pos_map = compute_pos_map_with_title_lowercased(raw_text)
#         except Exception as err:
#             print(f"[WARN] POS map error, empty map used: {err}", file=sys.stderr)
#             pos_map = {}

#         # c) pre-extraction POS filter (opsional → default True di config kamu)
#         source_text = (
#             build_pos_filtered_text(text_for_pos, settings.allowed_part_of_speech_tags)
#             if settings.enable_pre_extraction_pos_filter
#             else text_for_pos
#         )

#         # d) teks YAKE (emoji sudah bersih; flatten whitespace; lowercase)
#         yake_text = to_yake_source_from_pos(source_text)

#         # e) YAKE top-K awal
#         initial_pairs = yake_topk_pairs(yake_text, yake_extractor, settings.yake_initial_top_k)
#         keyword_pairs_all.append(initial_pairs)

#         # f) Rule A/B → final top-K
#         kept_pairs = postprocess_rule_ab(
#             initial_pairs,
#             final_top_k=settings.yake_final_top_k,
#             drop_all_stopwords_phrases=True,
#             custom_stopwords_iterable=custom_stopwords,
#         )

#         # g) timing
#         time_ms_list.append((perf_counter() - t0) * 1000.0)

#         # h) persist + POS audit (pakai most_frequent_tag)
#         final_phrases = [p for p, _ in kept_pairs]
#         keywords_final.append(final_phrases)
#         keyword_pairs_final.append(kept_pairs)

#         pos_audit = [
#             {"phrase": p, "pos": [most_frequent_tag(pos_map.get(tok.lower(), [])) for tok in toks(p)]}
#             for p in final_phrases
#         ]
#         keywords_with_pos.append(pos_audit)

#     # 4) tulis kolom hasil
#     df["keywords"] = keywords_final
#     df["keywords_str"] = df["keywords"].apply(lambda xs: ", ".join(xs) if xs else "")
#     if settings.keep_scores_column:
#         df["keywords_scored"] = [json.dumps(pairs, ensure_ascii=False) for pairs in keyword_pairs_final]
#     df["keywords_scored_all"] = [json.dumps(pairs, ensure_ascii=False) for pairs in keyword_pairs_all]
#     df["keywords_with_pos"] = [json.dumps(records, ensure_ascii=False) for records in keywords_with_pos]
#     df["time_to_5_keywords_ms_total"] = time_ms_list

#     # 5) save
#     Path(settings.output_csv).parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(settings.output_csv, index=False, encoding="utf-8")
#     print(f"[OK] Saved → {settings.output_csv}")
#     print(df[["keywords_str", "keywords_with_pos", "keywords_scored_all"]].head(3))

#     # ringkas timing
#     try:
#         print(
#             "[TIMING total] mean = {:.1f} ms | median = {:.1f} ms | max = {:.1f} ms | min = {:.1f} ms".format(
#                 df["time_to_5_keywords_ms_total"].mean(),
#                 df["time_to_5_keywords_ms_total"].median(),
#                 df["time_to_5_keywords_ms_total"].max(),
#                 df["time_to_5_keywords_ms_total"].min(),
#             )
#         )
#     except Exception:
#         pass


# if __name__ == "__main__":
#     run_pipeline()
