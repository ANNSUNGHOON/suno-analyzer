"""
Genre Taxonomy — Essentia ↔ 표준 장르 매핑 및 정규화 엔진
Multi-source consensus architecture의 기반 모듈

Essentia의 mtg-jamendo-genre 모델은 87개 장르 태그를 출력하고,
discogs-effnet은 400+ 스타일을 출력한다.
이 모듈은 이들을 정규화하여 Gemini Flash 출력과 비교 가능하게 만든다.
"""

from __future__ import annotations
from typing import Optional

# ============================================================
# MTG-Jamendo Genre 태그 (87개) → 표준 정규화 카테고리 매핑
# 출처: essentia.upf.edu/models.html (mtg_jamendo_genre)
# ============================================================
JAMENDO_NORMALIZED = {
    # --- Electronic ---
    "electronic": "electronic",
    "techno": "techno",
    "house": "house",
    "trance": "trance",
    "ambient": "ambient",
    "chillout": "chillout",
    "downtempo": "downtempo",
    "drumandbass": "drum and bass",
    "dubstep": "dubstep",
    "triphop": "trip-hop",
    "idm": "idm",
    "electropop": "electropop",
    "synthpop": "synth-pop",
    "edm": "edm",
    "deephouse": "deep house",
    "progressive": "progressive",
    # --- Rock ---
    "rock": "rock",
    "alternativerock": "alternative rock",
    "indierock": "indie rock",
    "hardrock": "hard rock",
    "punkrock": "punk rock",
    "postrock": "post-rock",
    "grunge": "grunge",
    "psychedelic": "psychedelic rock",
    "progressiverock": "progressive rock",
    "garagerock": "garage rock",
    "stonerrock": "stoner rock",
    # --- Metal ---
    "metal": "metal",
    "heavymetal": "heavy metal",
    "deathmetal": "death metal",
    "blackmetal": "black metal",
    "thrashmetal": "thrash metal",
    "metalcore": "metalcore",
    "doommetal": "doom metal",
    "numetal": "nu-metal",
    "industrial": "industrial",
    "deathcore": "deathcore",
    # --- Pop ---
    "pop": "pop",
    "indiepop": "indie pop",
    "synthwave": "synthwave",
    "disco": "disco",
    "dance": "dance pop",
    "funk": "funk",
    "soul": "soul",
    "rnb": "r&b",
    "kpop": "k-pop",
    "jpop": "j-pop",
    # --- Hip-Hop ---
    "hiphop": "hip-hop",
    "rap": "rap",
    "trap": "trap",
    "lofi": "lo-fi hip-hop",
    "boom bap": "boom bap",
    # --- Jazz ---
    "jazz": "jazz",
    "smoothjazz": "smooth jazz",
    "fusion": "jazz fusion",
    "bebop": "bebop",
    "swing": "swing",
    "bossa": "bossa nova",
    "bigband": "big band",
    # --- Classical ---
    "classical": "classical",
    "orchestra": "orchestral",
    "orchestral": "orchestral",
    "chamber": "chamber music",
    "baroque": "baroque",
    "romantic": "romantic classical",
    "contemporary": "contemporary classical",
    "choral": "choral",
    "opera": "opera",
    "soundtrack": "soundtrack",
    "cinematic": "cinematic",
    # --- Folk / Country / World ---
    "folk": "folk",
    "country": "country",
    "bluegrass": "bluegrass",
    "celtic": "celtic",
    "latin": "latin",
    "reggae": "reggae",
    "ska": "ska",
    "african": "afrobeat",
    "world": "world music",
    "flamenco": "flamenco",
    # --- Blues / R&B ---
    "blues": "blues",
    "gospel": "gospel",
    # --- Other ---
    "newage": "new age",
    "experimental": "experimental",
    "noise": "noise",
    "drone": "drone",
    "spoken": "spoken word",
    "acoustic": "acoustic",
    "singer-songwriter": "singer-songwriter",
    "easy listening": "easy listening",
    "lounge": "lounge",
}

# ============================================================
# Discogs 400+ 스타일 → 상위 카테고리 매핑
# Discogs top-level genres만 추출 (세부 스타일은 하위 매핑)
# ============================================================
DISCOGS_TOP_GENRES = {
    "Rock", "Electronic", "Pop", "Hip Hop", "Jazz", "Classical",
    "Folk, World, & Country", "Reggae", "Latin", "Blues",
    "Funk / Soul", "Stage & Screen", "Non-Music", "Brass & Military",
    "Children's",
}

# Discogs top-level → 표준 카테고리
DISCOGS_NORMALIZED = {
    "Rock": "rock",
    "Electronic": "electronic",
    "Pop": "pop",
    "Hip Hop": "hip-hop",
    "Jazz": "jazz",
    "Classical": "classical",
    "Folk, World, & Country": "folk",
    "Reggae": "reggae",
    "Latin": "latin",
    "Blues": "blues",
    "Funk / Soul": "funk",
    "Stage & Screen": "soundtrack",
    "Non-Music": "spoken word",
    "Brass & Military": "orchestral",
    "Children's": "pop",
}

# ============================================================
# 상위 카테고리 그룹 (fuzzy matching용)
# 같은 그룹에 속하면 "근접 일치"로 처리
# ============================================================
GENRE_FAMILY = {
    "electronic": {"electronic", "techno", "house", "trance", "ambient",
                   "chillout", "downtempo", "drum and bass", "dubstep",
                   "trip-hop", "idm", "electropop", "synth-pop", "edm",
                   "deep house", "progressive", "synthwave"},
    "rock": {"rock", "alternative rock", "indie rock", "hard rock",
             "punk rock", "post-rock", "grunge", "psychedelic rock",
             "progressive rock", "garage rock", "stoner rock"},
    "metal": {"metal", "heavy metal", "death metal", "black metal",
              "thrash metal", "metalcore", "doom metal", "nu-metal",
              "industrial", "deathcore"},
    "pop": {"pop", "indie pop", "dance pop", "disco", "k-pop", "j-pop",
            "electropop", "synth-pop"},
    "hip-hop": {"hip-hop", "rap", "trap", "lo-fi hip-hop", "boom bap"},
    "jazz": {"jazz", "smooth jazz", "jazz fusion", "bebop", "swing",
             "bossa nova", "big band"},
    "classical": {"classical", "orchestral", "chamber music", "baroque",
                  "romantic classical", "contemporary classical", "choral",
                  "opera", "soundtrack", "cinematic"},
    "folk": {"folk", "country", "bluegrass", "celtic", "world music",
             "flamenco", "afrobeat"},
    "blues": {"blues", "r&b", "soul", "gospel", "funk"},
}

# 역방향 lookup: genre → family
_GENRE_TO_FAMILY: dict[str, str] = {}
for family, members in GENRE_FAMILY.items():
    for g in members:
        _GENRE_TO_FAMILY[g] = family


# ============================================================
# Slug 정규화 — 공백/언더스코어/하이픈/약어 통일
# ============================================================
GENRE_ALIASES: dict[str, str] = {
    "dnb": "drum and bass",
    "d&b": "drum and bass",
    "dandb": "drum and bass",
    "d'n'b": "drum and bass",
    "drum&bass": "drum and bass",
    "drum'n'bass": "drum and bass",
    "hh": "hip-hop",
    "hiphop": "hip-hop",
    "r&b": "r&b",
    "rnb": "r&b",
    "rhythmandblues": "r&b",
    "kpop": "k-pop",
    "jpop": "j-pop",
    "synthpop": "synth-pop",
    "triphop": "trip-hop",
    "postrock": "post-rock",
    "numetal": "nu-metal",
    "lofi": "lo-fi hip-hop",
    "lo-fi": "lo-fi hip-hop",
    "bossanova": "bossa nova",
    "bigband": "big band",
    "deephouse": "deep house",
    "hardrock": "hard rock",
    "indierock": "indie rock",
    "indiepop": "indie pop",
    "punkrock": "punk rock",
    "altrock": "alternative rock",
    "progrock": "progressive rock",
    "blackmetal": "black metal",
    "deathmetal": "death metal",
    "heavymetal": "heavy metal",
    "thrashmetal": "thrash metal",
    "doommetal": "doom metal",
    "stonerrock": "stoner rock",
    "garagerock": "garage rock",
    "smoothjazz": "smooth jazz",
    "jazzfusion": "jazz fusion",
    "contemporaryclassical": "contemporary classical",
    "chambermusic": "chamber music",
    "worldmusic": "world music",
    "dancepop": "dance pop",
    "electropop": "electropop",
    "drumandbass": "drum and bass",
    "psychedelicrock": "psychedelic rock",
    "progressiverock": "progressive rock",
    "alternativerock": "alternative rock",
    "easylistening": "easy listening",
    "newage": "new age",
    "spokenword": "spoken word",
    "singersongwriter": "singer-songwriter",
    "romanticclassical": "romantic classical",
}


def _slugify(raw: str) -> str:
    """장르 문자열을 비교용 슬러그로 변환.

    공백, 언더스코어, 하이픈, 슬래시, 앰퍼샌드, 마침표를
    모두 제거하여 단일 소문자 문자열로 통일.

    "black metal" → "blackmetal"
    "black_metal" → "blackmetal"
    "black-metal" → "blackmetal"
    "drum and bass" → "drumandbass"
    "drum_and_bass" → "drumandbass"
    """
    s = raw.strip().lower()
    s = s.replace("&", "and")  # & → and (drum & bass → drumandbass)
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    s = s.replace("/", "").replace(".", "").replace("'", "")
    return s


# ============================================================
# 정규화 함수
# ============================================================

def normalize_genre(raw: str) -> str:
    """원시 장르 문자열을 표준 카테고리로 정규화.

    순서:
    0. GENRE_ALIASES 약어/변형 매칭 (dnb, rnb, kpop 등)
    1. JAMENDO_NORMALIZED slug 매칭
    2. DISCOGS_NORMALIZED 정확 매칭
    3. 부분 문자열 매칭 (가장 긴 매치 우선)
    4. 못 찾으면 소문자 strip 반환

    _slugify()로 공백/언더스코어/하이픈 차이를 무시하므로
    "black metal" = "black_metal" = "black-metal" 모두 동일.
    """
    if not raw:
        return ""

    clean = raw.strip().lower()
    slug = _slugify(raw)

    # 0. 약어/변형 매칭 (dnb → drum and bass 등)
    if slug in GENRE_ALIASES:
        return GENRE_ALIASES[slug]

    # 1. Jamendo slug 매칭
    if slug in JAMENDO_NORMALIZED:
        return JAMENDO_NORMALIZED[slug]

    # 1b. 원본 소문자로도 시도
    if clean in JAMENDO_NORMALIZED:
        return JAMENDO_NORMALIZED[clean]

    # 2. 부분 문자열 매칭 — slug 기준 (긴 것 우선, 더 구체적)
    best_key = ""
    best_val = ""
    for key, normalized in JAMENDO_NORMALIZED.items():
        if key in slug and len(key) > len(best_key):
            best_key = key
            best_val = normalized
    if best_val:
        return best_val

    # 3. Discogs top-level 매칭
    for dg, normalized in DISCOGS_NORMALIZED.items():
        if _slugify(dg) == slug:
            return normalized

    # 4. 원본 반환
    return clean


def get_genre_family(genre: str) -> Optional[str]:
    """장르의 상위 패밀리를 반환. 없으면 None."""
    normalized = normalize_genre(genre)
    return _GENRE_TO_FAMILY.get(normalized)


def genres_match(genre_a: str, genre_b: str, strict: bool = False) -> bool:
    """두 장르가 일치하는지 판단.

    strict=True: 정규화된 문자열 완전 일치만
    strict=False: 같은 패밀리에 속하면 일치로 인정

    _slugify() 기반이므로 공백/언더스코어/하이픈 차이 무시.
    """
    norm_a = normalize_genre(genre_a)
    norm_b = normalize_genre(genre_b)

    # 정확 일치 (정규화 후)
    if norm_a == norm_b:
        return True

    # slug 비교 (정규화 결과의 slug도 비교)
    if _slugify(norm_a) == _slugify(norm_b):
        return True

    if strict:
        return False

    # 패밀리 일치
    fam_a = _GENRE_TO_FAMILY.get(norm_a)
    fam_b = _GENRE_TO_FAMILY.get(norm_b)

    if fam_a and fam_b and fam_a == fam_b:
        return True

    # 한쪽이 다른 쪽의 부분 문자열 (slug 기준)
    slug_a = _slugify(norm_a)
    slug_b = _slugify(norm_b)
    if slug_a in slug_b or slug_b in slug_a:
        return True

    return False


def match_strength(genre_a: str, genre_b: str) -> str:
    """매칭 강도를 반환: "exact" | "family" | "partial" | "none"

    _slugify() 기반이므로 "black metal" vs "black_metal" → exact.
    """
    norm_a = normalize_genre(genre_a)
    norm_b = normalize_genre(genre_b)

    # 정규화 후 일치 또는 slug 일치 → exact
    if norm_a == norm_b or _slugify(norm_a) == _slugify(norm_b):
        return "exact"

    fam_a = _GENRE_TO_FAMILY.get(norm_a)
    fam_b = _GENRE_TO_FAMILY.get(norm_b)
    if fam_a and fam_b and fam_a == fam_b:
        return "family"

    slug_a = _slugify(norm_a)
    slug_b = _slugify(norm_b)
    if slug_a in slug_b or slug_b in slug_a:
        return "partial"

    return "none"