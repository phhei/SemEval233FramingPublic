LANGUAGE_ABBREVIATION_TO_FULL = {
    "en": "english",
    "fr": "french",
    "it": "italian",
    "ru": "russian",
    "po": "polish",
    "ge": "german",
    "es": "spanish",
    "gr": "greek",
    "ka": "georgian",
    "es2en": "english",
    "fr2en": "english",
    "ge2en": "english",
    "gr2en": "english",
    "it2en": "english",
    "ka2en": "english",
    "po2en": "english",
    "ru2en": "english"
}
LANGUAGE_FULL_TO_ABBREVIATION = {
    "english": "en",
    "french": "fr",
    "italian": "it",
    "russian": "ru",
    "polish": "po",
    "german": "ge",
    "spanish": "es",
    "greek": "gr",
    "georgian": "ka"
}


def get_pure_language_abbreviation(language_tag: str) -> str:
    if "2" in language_tag:
        return language_tag.split(sep="2")[-1]

    return language_tag