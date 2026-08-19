"""Medical facts for Merge — verified, never invented, and NOT a doctor.

The mission is communication, both directions:
  * turn a patient's plain words into the PRECISE clinical term the doctor
    can't misread ("sugar problem" -> "type 2 diabetes mellitus, ICD-10 E11"),
  * turn the doctor's jargon back into plain language the patient understands.

Every fact is grounded in an authoritative source (NLM RxNorm/MedlinePlus/
ClinicalTables, FDA openFDA). A drug or condition that doesn't verify is
reported as NOT FOUND, never guessed. Nothing here diagnoses, doses, or
advises treatment — those belong to a licensed clinician, and emergencies
belong to 911. Same architecture as the legal tools: mechanical grounding,
hard safety lines, honest scope.
"""
from __future__ import annotations

import html
import re

import httpx

UA = {"User-Agent": "Mozilla/5.0 (research)"}
RXNAV = "https://rxnav.nlm.nih.gov/REST"
OPENFDA = "https://api.fda.gov/drug/label.json"
CT = "https://clinicaltables.nlm.nih.gov/api"
MEDLINE = "https://wsearch.nlm.nih.gov/ws/query"

DISCLAIMER = (
    "Medical INFORMATION, not medical advice. Not a diagnosis, not a "
    "treatment or dosing recommendation, and not a substitute for a licensed "
    "clinician who can examine you. For an emergency, call 911 (US) or your "
    "local emergency number now.")


def _get(url: str, params: dict, kind: str = "json"):
    r = httpx.get(url, params=params, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json() if kind == "json" else r.text


def _strip(s: str) -> str:
    # Unescape FIRST, then remove tags: MedlinePlus double-escapes its markup
    # (&lt;span&gt;), so stripping before unescaping leaves the tags behind.
    # Only strip REAL html tags (start with a letter or /), so clinical text
    # like "keep below (<70) and above (>200)" survives instead of being eaten.
    return re.sub(r"</?[a-zA-Z][^<>]*>", "", html.unescape(s or "")).strip()


# --------------------------------------------------------------------------
def verify_drug(name: str) -> str:
    """Confirm a medication is real and give its PRECISE names, so the patient
    can tell the doctor/pharmacist the exact drug. VERIFIED with generic/brand
    names + class, or NOT FOUND. No dosing, no 'should you take it'."""
    name = (name or "").strip()
    if not name:
        return "CHECK FAILED — no drug name given. " + DISCLAIMER
    try:
        d = _get(f"{RXNAV}/rxcui.json", {"name": name})
        ids = (d.get("idGroup") or {}).get("rxnormId") or []
        if not ids:
            # try an approximate match to catch misspellings, but never assert
            appr = _get(f"{RXNAV}/approximateTerm.json",
                        {"term": name, "maxEntries": 3})
            cands = (appr.get("approximateGroup") or {}).get("candidate") or []
            near = sorted({c.get("name", "") for c in cands if c.get("name")})[:3]
            hint = (f" Did you mean: {', '.join(near)}? Verify the exact one."
                    if near else "")
            return (f"NOT FOUND — no medication named '{name}' matched RxNorm "
                    f"(the US drug terminology).{hint} Don't cite it as a real "
                    f"drug or state what it does. " + DISCLAIMER)
        rxcui = ids[0]
    except Exception as e:
        return (f"CHECK FAILED — couldn't reach the drug database "
                f"({type(e).__name__}). Nothing verified; don't state drug "
                f"facts from memory. " + DISCLAIMER)

    # Names + class straight from RxNorm/RxClass (authoritative, one source).
    generic, brand, klass = "", "", ""
    try:
        rel = _get(f"{RXNAV}/rxcui/{rxcui}/related.json", {"tty": "IN BN"})
        for g in (rel.get("relatedGroup") or {}).get("conceptGroup", []):
            names = [c["name"] for c in (g.get("conceptProperties") or [])]
            if g.get("tty") == "IN" and names:
                generic = "; ".join(sorted(set(names)))[:120]
            elif g.get("tty") == "BN" and names:
                brand = "; ".join(sorted(set(names))[:6])[:140]
    except Exception:
        pass
    try:
        cl = _get(f"{RXNAV}/rxclass/class/byRxcui.json",
                  {"rxcui": rxcui, "relaSource": "ATC"})
        cs = sorted({c.get("rxclassMinConceptItem", {}).get("className", "")
                     for c in (cl.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", [])
                     if c.get("rxclassMinConceptItem", {}).get("className")})
        klass = "; ".join(cs[:4])[:160]
    except Exception:
        pass

    lines = [f"VERIFIED — '{name}' is a real medication (RxNorm CUI {rxcui})."]
    if generic:
        lines.append(f"  Generic (clinical) name: {generic}")
    if brand:
        lines.append(f"  Sold under brand names like: {brand}")
    if klass:
        lines.append(f"  Drug class: {klass}")
    tell = ("the GENERIC name above" if generic
            else f"the exact name '{name}' and the RxNorm CUI {rxcui}")
    lines.append(f"  Tell the doctor/pharmacist {tell} — it's the unambiguous "
                 "identifier. Do NOT rely on me for dose, interactions, or "
                 "whether it's right for you; ask the pharmacist or prescriber.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# --------------------------------------------------------------------------
def find_condition(term: str) -> str:
    """Given plain or garbled words for a health problem, return the PRECISE
    clinical term(s) and ICD-10 code(s) the doctor uses — so the patient can
    say the exact thing and not be misread. NOT a diagnosis: these are the
    words for what was DESCRIBED, not a claim the patient has it."""
    term = (term or "").strip()
    if not term:
        return "CHECK FAILED — nothing to look up. " + DISCLAIMER
    meaningful = re.sub(r"[^a-zA-Z]+", " ", term).strip()
    if len(meaningful) < 3 or not any(len(w) >= 3 for w in meaningful.split()):
        return ("NOT FOUND — that doesn't look like a health term to describe. "
                "Say the problem in plain words (e.g. 'high blood sugar'). " + DISCLAIMER)
    try:
        icd = _get(f"{CT}/icd10cm/v3/search",
                   {"sf": "code,name", "terms": term, "maxList": 6})
        cond = _get(f"{CT}/conditions/v3/search",
                    {"terms": term, "maxList": 4})
    except Exception as e:
        return (f"CHECK FAILED — couldn't reach the terminology database "
                f"({type(e).__name__}). Don't invent a clinical term. "
                + DISCLAIMER)

    icd_hits = icd[3] if len(icd) > 3 else []
    cond_hits = cond[3] if len(cond) > 3 else []
    if not icd_hits and not cond_hits:
        return (f"NOT FOUND — nothing in the medical vocabularies matched "
                f"'{term}'. It may be misspelled or not a standard term; don't "
                f"put an official-sounding label on it. Describe it plainly to "
                f"the doctor instead. " + DISCLAIMER)

    lines = [f"Precise clinical wording for '{term}' (use these EXACT terms so "
             f"the doctor can't misread you — this is vocabulary, NOT a "
             f"diagnosis that you have any of them):"]
    if cond_hits:
        lines.append("  Common condition names:")
        for c in cond_hits[:4]:
            lines.append(f"    · {_strip(c[0])}")
    if icd_hits:
        lines.append("  Clinical terms with ICD-10 codes:")
        for c in icd_hits[:6]:
            lines.append(f"    · {c[1]}  [{c[0]}]")
    lines.append("Only a clinician can say which (if any) applies to you.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# --------------------------------------------------------------------------
def explain_plain(term: str) -> str:
    """Translate a medical term/topic into plain language (MedlinePlus, NLM's
    consumer health service). For turning the doctor's jargon into words the
    patient actually understands."""
    term = (term or "").strip()
    if not term:
        return "CHECK FAILED — nothing to explain. " + DISCLAIMER
    try:
        xml = _get(MEDLINE, {"db": "healthTopics", "term": term, "retmax": 1},
                   kind="text")
    except Exception as e:
        return (f"CHECK FAILED — couldn't reach MedlinePlus "
                f"({type(e).__name__}). " + DISCLAIMER)
    title = re.search(r'<content name="title">(.*?)</content>', xml, re.S)
    summ = re.search(r'<content name="FullSummary">(.*?)</content>', xml, re.S)
    snip = re.search(r'<content name="snippet">(.*?)</content>', xml, re.S)
    url = re.search(r"<url>(.*?)</url>", xml, re.S)
    if not title:
        return (f"No MedlinePlus plain-language topic matched '{term}'. Don't "
                f"improvise a definition; point them to their clinician or "
                f"medlineplus.gov. " + DISCLAIMER)
    body = _strip((summ or snip).group(1)) if (summ or snip) else ""
    body = (body[:700] + "…") if len(body) > 700 else body
    out = [f"Plain-language — {_strip(title.group(1))}:", body or "(summary "
           "unavailable; see the link)"]
    if url:
        out.append(f"Full plain-language page: {_strip(url.group(1))}")
    out.append(DISCLAIMER)
    return "\n".join(out)


def medical_notice() -> str:
    """The standing medical disclaimer."""
    return DISCLAIMER
