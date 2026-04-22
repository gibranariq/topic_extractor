from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized configuration for the YAKE keyword extraction pipeline.
    You can override these with environment variables if needed.
    """

    # I/O paths
    input_csv: str = "/Users/gibranariq/Documents/kumparan/final-project/data_test.csv"
    output_csv: str = "/Users/gibranariq/Documents/kumparan/final-project/output/ekstrak_coba_2.csv"
    text_column: str = "content"

    # Stopwords file (TXT, one entry per line; phrases allowed)
    stopwords_path: str = "/Users/gibranariq/Documents/kumparan/final-project/utils/stopwords.txt"

    # YAKE parameters
    yake_language: str = "id"
    yake_max_ngram: int = 2
    # yake_top_k: int = 5
    yake_deduplication_limit: float = 0.9  # 0..1 (smaller = more aggressive deduplication)
    yake_deduplication_function: str = "seqm"  # "seqm" or "jaccard"
    yake_window_size: int = 1  # small = tighter, phrase-level focus (good for news/SEO)

    # Candidate budgeting
    yake_initial_top_k: int = 50 # take more first, then post-process
    yake_final_top_k: int = 10   # final number of keywords to keep

    # Output options
    keep_scores_column: bool = True  # store [["keyword", score], ...] JSON for audit

    # Part-of-speech filtering (using nlp_id)
    # enable_part_of_speech_filter: bool = False
    enable_pre_extraction_pos_filter: bool = True
    allowed_part_of_speech_tags: set[str] = {"NN", "NP", "NNP", "JJ", "ADJP"}


    # Post-extraction POS filter toggle (this variant uses it implicitly in code)
    # enable_post_extraction_pos_filter: bool = True

settings = Settings()
