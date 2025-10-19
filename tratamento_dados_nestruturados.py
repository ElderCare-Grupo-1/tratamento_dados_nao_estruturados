import easyocr
import re
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process  # pip install rapidfuzz
import cv2
from unidecode import unidecode  # pip install Unidecode


def preprocess_image(path, scale=1.0):
    """Leitura e pré-processamento leve (opcional) para melhorar OCR.
    Retorna imagem em BGR ou lança FileNotFoundError.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # redução de ruído e equalização
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # binarização adaptativa (para texto)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
    # inverter para easyocr (preto texto sobre branco)
    th = cv2.bitwise_not(th)
    return th


def tokenize(text):
    # tokens contendo apenas letras (removendo números e símbolos)
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)


def tokenize_keep_numbers(text):
    # tokens contendo letras e números (preserva doses como 1G, 600MG)
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", text)


def normalize(s):
    s2 = unidecode(s or "").upper()
    # remove tudo que não seja letra (remove números e caracteres especiais)
    s2 = re.sub(r"[^A-Z]", "", s2)
    return s2


def normalize_keep_numbers(s):
    """Normaliza mantendo letras e números (preserva doses como 1G, 600MG)."""
    s2 = unidecode(s or "").upper()
    # mantém apenas letras A-Z e dígitos 0-9
    s2 = re.sub(r"[^A-Z0-9]", "", s2)
    return s2


def strip_dose(s: str) -> str:
    """Remove padrões de dose como '1G', '600MG', '10 ML' e números avulsos.

    Retorna a string sem unidades de dose — útil para comparar apenas o nome.
    """
    if not s:
        return ""
    t = unidecode(s).upper()
    # remover padrões comuns: número + unidade (MG, G, ML, MCG)
    t = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(MG|G|ML|MCG|MG/ML|MGML)\b", "", t)
    # remover números isolados
    t = re.sub(r"\b\d+(?:[\.,]\d+)?\b", "", t)
    # remover pontuação sobrante
    t = re.sub(r"[^A-Z\s]", "", t)
    # compactar espaços e strip
    t = re.sub(r"\s+", " ", t).strip()
    return t


def generate_ngrams(tokens, max_n=3):
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i + n]))
    return ngrams


# module-level defaults
DEFAULT_STOPWORDS = {"DATA", "NOME", "HORA", "DRA", "DAS", "DOS", "DE", "E"}


def load_base(base_csv: str):
    """Carrega e normaliza a base de remédios.

    Retorna: (base_norm_map, base_norm_map_num, base_norm_list, base_norm_list_num, base_word_set)
    """
    base_df = pd.read_csv(base_csv, dtype=str)
    base_list = base_df['NOME_PRODUTO'].fillna("").astype(str).tolist()
    base_norm_map = {normalize(name): name for name in base_list if normalize(name)}
    base_norm_list = list(base_norm_map.keys())
    base_norm_map_num = {normalize_keep_numbers(name): name for name in base_list if normalize_keep_numbers(name)}
    base_norm_list_num = list(base_norm_map_num.keys())
    # versão da base sem doses (remove 1G, 600MG etc) para matching mais permissivo
    base_norm_map_nodose = {normalize(strip_dose(name)): name for name in base_list if normalize(strip_dose(name))}
    base_norm_list_nodose = list(base_norm_map_nodose.keys())

    base_word_set = set()
    for name in base_list:
        toks = re.findall(r"[A-Za-z0-9]+", name)
        for t in toks:
            tn = normalize(t)
            if tn:
                base_word_set.add(tn)

    return base_norm_map, base_norm_map_num, base_norm_map_nodose, base_norm_list, base_norm_list_num, base_norm_list_nodose, base_word_set


def extract_ocr_lines(reader, image_path: str, preprocess: bool):
    """Roda o OCR (com ou sem pré-processamento) e normaliza o retorno em [(text, conf)]."""
    if preprocess:
        img = preprocess_image(image_path)
        ocr_results = reader.readtext(img, detail=1)
    else:
        ocr_results = reader.readtext(image_path, detail=1)

    lines: List[Tuple[str, float]] = []
    for item in ocr_results:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            _, text, conf = item
        else:
            text = str(item)
            conf = 0.0
        text = (text or "").strip().upper()
        if not text:
            continue
        raw_conf = float(conf)
        if raw_conf > 1.0:
            raw_conf = min(1.0, raw_conf / 100.0)
        ocr_conf = max(0.0, min(1.0, raw_conf))
        lines.append((text, ocr_conf))
    return lines


def build_candidates_from_lines(lines: List[Tuple[str, float]], ngram_max: int = 3):
    """Gera tokens (preservando números) e n-grams com confiança média."""
    full_text = " ".join([t for t, _ in lines])
    tokens = tokenize_keep_numbers(full_text)

    token_confs: List[float] = []
    for text, conf in lines:
        toks = tokenize_keep_numbers(text)
        if not toks:
            continue
        token_confs.extend([conf] * len(toks))
    if len(token_confs) < len(tokens):
        token_confs.extend([0.5] * (len(tokens) - len(token_confs)))

    candidates: Dict[str, float] = {}
    for n in range(1, ngram_max + 1):
        for i in range(len(tokens) - n + 1):
            ng = " ".join(tokens[i:i + n])
            confs = token_confs[i:i + n]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            candidates[ng] = max(candidates.get(ng, 0.0), avg_conf)

    return candidates


def match_candidate(cand: str,
                    ocr_conf: float,
                    base_norm_list: List[str],
                    base_norm_list_num: List[str],
                    base_norm_map: Dict[str, str],
                    base_norm_map_num: Dict[str, str],
                    base_norm_map_nodose: Dict[str, str],
                    base_norm_list_nodose: List[str],
                    base_word_set: set,
                    fuzzy_threshold: int,
                    combined_threshold: int,
                    ocr_weight: float,
                    min_token_len: int,
                    min_candidate_len: int,
                    stopwords: Optional[set],
                    debug: bool = False) -> Optional[Dict]:
    """Tentativa de casar um candidato com a base. Retorna dict com detalhes ou None."""
    cand_norm_num = normalize_keep_numbers(cand)
    cand_norm = normalize(cand)
    if not cand_norm and not cand_norm_num:
        return None

    token_candidates = re.findall(r"[A-Z0-9]+", cand_norm_num if cand_norm_num else cand_norm)
    token_letters = [re.sub(r"[^A-Z]", "", t) for t in token_candidates]
    token_letters = [t for t in token_letters if len(t) >= min_token_len]
    ok_token = any(t in base_word_set for t in token_letters)
    ok_sub = any(bn.find(cand_norm) != -1 for bn in base_norm_list) or any(bn.find(cand_norm_num) != -1 for bn in base_norm_list_num)

    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    if cand_norm in stopwords:
        if debug:
            print(f"SKIP stopword candidate: {cand} -> {cand_norm}")
        return None

    if len(cand_norm) < min_candidate_len and not ok_token and not ok_sub and cand_norm not in base_norm_list:
        if debug:
            print(f"SKIP short candidate: {cand} (len={len(cand_norm)}) tokens={token_candidates} ok_token={ok_token} ok_sub={ok_sub}")
        return None

    if not (cand_norm in base_norm_list or ok_sub or ok_token):
        return None

    # Primeiro tente matching preservando números
    best = None
    matched_norm = None
    fuzzy_score = 0.0
    orig_name = cand
    if cand_norm_num:
        best_num = process.extractOne(cand_norm_num, base_norm_list_num, scorer=fuzz.WRatio)
        if best_num and best_num[1] >= fuzzy_threshold:
            best = best_num
            matched_norm = best[0]
            fuzzy_score = float(best[1])
            orig_name = base_norm_map_num.get(matched_norm, matched_norm)

    # fallback sem números
    if best is None:
        best = process.extractOne(cand_norm, base_norm_list, scorer=fuzz.WRatio)
        if not best:
            # tentar matching sem dose (ex: "LISADOR DIP 1G" -> "LISADOR DIP")
            cand_nodose = normalize(strip_dose(cand))
            if cand_nodose:
                best_n = process.extractOne(cand_nodose, base_norm_list_nodose, scorer=fuzz.WRatio)
                if best_n and best_n[1] >= fuzzy_threshold:
                    matched_norm = best_n[0]
                    fuzzy_score = float(best_n[1])
                    orig_name = base_norm_map_nodose.get(matched_norm, matched_norm)
                else:
                    return None
            else:
                return None
        matched_norm = best[0]
        fuzzy_score = float(best[1])
        orig_name = base_norm_map.get(matched_norm, matched_norm)

    combined = fuzzy_score * ((1.0 - ocr_weight) + ocr_weight * float(ocr_conf))
    valid = (fuzzy_score >= fuzzy_threshold) and (combined >= combined_threshold)

    return {
        "candidate": cand,
        "candidate_norm": cand_norm,
        "matched_norm": matched_norm,
        "matched_name": orig_name,
        "fuzzy": fuzzy_score,
        "ocr_conf": float(ocr_conf),
        "combined": float(combined),
        "valid": bool(valid)
    }


def ocr_and_match(image_path,
                  base_csv="remedios_base_tratada.csv",
                  fuzzy_threshold=80,
                  combined_threshold=75,
                  ngram_max=3,
                  preprocess=True,
                  debug=False,
                  ocr_weight=0.2,
                  min_candidate_len=4,
                  min_token_len=3,
                  stopwords=None):
    # Inicializa OCR (pt + en)
    reader = easyocr.Reader(["pt", "en"], gpu=False)

    # carregar base e linhas OCR
    base_norm_map, base_norm_map_num, base_norm_map_nodose, base_norm_list, base_norm_list_num, base_norm_list_nodose, base_word_set = load_base(base_csv)
    lines = extract_ocr_lines(reader, image_path, preprocess)

    if debug:
        print("\n--- OCR lines (text, conf) ---")
        for t, c in lines:
            print(f"{c:.2f}\t{t}")

    candidates = build_candidates_from_lines(lines, ngram_max=ngram_max)

    if debug:
        example_tokens = list(candidates.keys())[:50]
        print("\nTokens/candidates (sample):", example_tokens)

    matched_details = []
    for cand, ocr_conf in candidates.items():
        md = match_candidate(
            cand=cand,
            ocr_conf=ocr_conf,
            base_norm_list=base_norm_list,
            base_norm_list_num=base_norm_list_num,
            base_norm_map=base_norm_map,
            base_norm_map_num=base_norm_map_num,
            base_norm_map_nodose=base_norm_map_nodose,
            base_norm_list_nodose=base_norm_list_nodose,
            base_word_set=base_word_set,
            fuzzy_threshold=fuzzy_threshold,
            combined_threshold=combined_threshold,
            ocr_weight=ocr_weight,
            min_token_len=min_token_len,
            min_candidate_len=min_candidate_len,
            stopwords=stopwords,
            debug=debug
        )
        if md:
            matched_details.append(md)

    if debug:
        # show top candidate matches regardless of valid flag to help diagnose
        debug_df = pd.DataFrame(matched_details).sort_values(by='combined', ascending=False)
        print(f"\n--- Top {min(20, len(debug_df))} candidate matches (debug) ---")
        if not debug_df.empty:
            print(debug_df.head(20).to_string(index=False))
        else:
            print("(nenhum candidato gerado)")

    # dedupe by matched_name keeping highest combined
    best_per_name = {}
    for d in matched_details:
        name = d['matched_name']
        if name not in best_per_name or d['combined'] > best_per_name[name]['combined']:
            best_per_name[name] = d

    # prepare sorted results (only those marked valid)
    results_sorted = sorted([v for v in best_per_name.values() if v['valid']], key=lambda x: x['combined'], reverse=True)

    # save detailed CSV with all candidates and validity flag
    details_df = pd.DataFrame(matched_details)
    if not details_df.empty:
        details_df = details_df.sort_values(by=['combined'], ascending=False)
        details_df.to_csv("remedios_validados_detalhado.csv", index=False, encoding='utf-8-sig')

    # save only validated names
    validated = [d['matched_name'] for d in results_sorted]
    out_df = pd.DataFrame(validated, columns=["NOME_PRODUTO"]) if validated else pd.DataFrame(columns=["NOME_PRODUTO"])
    out_df.to_csv("remedios_validados.csv", index=False, encoding='utf-8-sig')

    return results_sorted, matched_details


if __name__ == '__main__':
    IMAGE_PATH = r"D:\Faculdade\Captura de tela 2025-09-29 210530.png"
    print("Iniciando OCR e validação (modo debug)...")
    try:
        results_sorted, matched_details = ocr_and_match(
            IMAGE_PATH,
            base_csv="remedios_base_tratada.csv",
            fuzzy_threshold=60,
            combined_threshold=40,
            ngram_max=3,
            preprocess=True,
            debug=False,
            ocr_weight=0.2
        )
    except FileNotFoundError as e:
        print("Arquivo de imagem não encontrado:", e)
        raise
    except Exception as e:
        print("Erro durante OCR e matching:", e)
        raise

    print("Remédios validados:")
    # Mostrar somente resultados com combined > 90
    high_conf = [d for d in results_sorted if d.get('combined', 0) > 90]
    print("Remédios com combined > 90:")
    if high_conf:
        for d in high_conf:
            print(f"- {d['matched_name']} (combined={d['combined']:.1f}, fuzzy={d['fuzzy']:.1f}, ocr_conf={d['ocr_conf']:.2f}, cand='{d['candidate']}')")
    else:
        print("(nenhum resultado com combined > 90)")

    print(f"\nTotal candidatos processados: {len(matched_details)}")
    print(f"Total validados (combined>90): {len(high_conf)}")
