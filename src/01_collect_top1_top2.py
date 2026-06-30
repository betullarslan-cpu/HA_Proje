"""
Cosmos-T1 ile GSM8K-TR cevap uretimi ve token bazli top1-top2 fark kaydi.

Bu dosya Colab GPU ortaminda calistirilmak uzere hazirlanmistir.
Model yerel CPU'da pratik surelerde calismaz.
"""

import json
import os
import random
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList


MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-T1"
ROOT = "/content/drive/MyDrive/HA_but"
KAYIT_DOSYASI = f"{ROOT}/cosmos_t1_kayitlar_768.jsonl"

HEDEF_DOGRU = 550
HEDEF_YANLIS = 550
N_RUN = 2000
KAYIT_ARALIGI = 10
MAX_NEW_TOKENS = 1024

random.seed(42)


def sayi_temizle(s):
    s = str(s).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if s == "":
        return None

    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) == 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts[-1]) == 3 and len(parts) > 1:
            s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None


def gold_cevap(text):
    text = str(text)

    m = re.search(r"####\s*(-?\d+(?:[.,]\d+)*)", text)
    if m:
        return sayi_temizle(m.group(1))

    satirlar = [s.strip() for s in text.strip().split("\n") if s.strip()]
    for satir in reversed(satirlar):
        if re.search(r"\b\d{1,2}:\d{2}\b", satir):
            return None

        sayilar = re.findall(r"-?\d+(?:[.,]\d+)*", satir)
        if sayilar:
            return sayi_temizle(sayilar[-1])

    return None


def pred_cevap(text):
    text = str(text)

    m = re.search(r"####\s*(-?\d+(?:[.,]\d+)*)", text)
    if m:
        return sayi_temizle(m.group(1))

    m = re.search(
        r"(?:cevap|sonuç|sonuc|nihai cevap)[^\d\-]*(-?\d+(?:[.,]\d+)*)",
        text.lower(),
    )
    if m:
        return sayi_temizle(m.group(1))

    sayilar = re.findall(r"-?\d+(?:[.,]\d+)*", text)
    if sayilar:
        return sayi_temizle(sayilar[-1])

    return None


class Top1Top2Kaydedici(LogitsProcessor):
    def __init__(self):
        self.farklar = []

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float(), dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        fark = (top2[:, 0] - top2[:, 1]).detach().cpu().tolist()
        self.farklar.append(fark)
        return scores


def prompt_hazirla(soru):
    return (
        "Sadece kısa cevap ver. "
        "Adım adım uzun çözüm yazma. "
        "Son satır: #### <sayı>\n\n"
        f"Soru: {soru}"
    )


def kayitlari_oku(path):
    if not os.path.exists(path):
        return []

    kayitlar = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kayitlar.append(json.loads(line))
    return kayitlar


def kayitlari_yaz(path, yeni_kayitlar):
    if not yeni_kayitlar:
        return

    with open(path, "a", encoding="utf-8") as f:
        for kayit in yeni_kayitlar:
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


def veri_yukle():
    ds = load_dataset("ytu-ce-cosmos/gsm8k_tr")["train"]
    sorular = []

    for i, ex in enumerate(ds):
        if re.search(r"\b\d{1,2}:\d{2}\b", str(ex["answer"])):
            continue

        gold = gold_cevap(ex["answer"])
        if gold is None:
            continue

        sorular.append(
            {
                "id": i,
                "soru": ex["question"].strip(),
                "gold": gold,
                "gold_text": ex["answer"],
            }
        )

    random.shuffle(sorular)
    return sorular


def model_yukle():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    terminators = [tokenizer.eos_token_id]
    end_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_turn, int) and end_turn >= 0:
        terminators.append(end_turn)

    return tokenizer, model, list(set(terminators))


@torch.no_grad()
def uret(soru, tokenizer, model, terminators):
    messages = [{"role": "user", "content": prompt_hazirla(soru)}]
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    kaydedici = Top1Top2Kaydedici()

    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        logits_processor=LogitsProcessorList([kaydedici]),
        output_scores=False,
        return_dict_in_generate=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0][prompt_len:]
    metin = tokenizer.decode(gen_ids, skip_special_tokens=True)
    diffs = [float(x[0]) for x in kaydedici.farklar]

    del inputs, out, gen_ids
    torch.cuda.empty_cache()

    return metin, diffs


def main():
    os.makedirs(ROOT, exist_ok=True)

    sorular = veri_yukle()
    tokenizer, model, terminators = model_yukle()

    kayitlar = kayitlari_oku(KAYIT_DOSYASI)
    islenen_idler = set(r["id"] for r in kayitlar)
    dogru_sayisi = sum(r["dogru"] for r in kayitlar)
    yanlis_sayisi = len(kayitlar) - dogru_sayisi
    bekleyen_kayitlar = []

    print("Onceki kayit:", len(kayitlar))
    print("Dogru:", dogru_sayisi, "Yanlis:", yanlis_sayisi)

    basla = time.time()

    for q in sorular[:N_RUN]:
        if q["id"] in islenen_idler:
            continue

        metin, diffs = uret(q["soru"], tokenizer, model, terminators)
        pred = pred_cevap(metin)
        dogru = pred is not None and abs(pred - q["gold"]) < 1e-4

        kayit = {
            "id": q["id"],
            "soru": q["soru"],
            "gold": q["gold"],
            "pred": pred,
            "dogru": dogru,
            "diffs": diffs,
            "n_token": len(diffs),
            "metin_kismi": metin[:500],
            "metin_son": metin[-500:],
        }

        bekleyen_kayitlar.append(kayit)
        islenen_idler.add(q["id"])
        dogru_sayisi += int(dogru)
        yanlis_sayisi += int(not dogru)

        print(
            f"[{len(islenen_idler)}] gold={q['gold']} pred={pred} "
            f"{'DOGRU' if dogru else 'YANLIS'} token={len(diffs)} | "
            f"D={dogru_sayisi} Y={yanlis_sayisi}"
        )

        if len(bekleyen_kayitlar) >= KAYIT_ARALIGI:
            kayitlari_yaz(KAYIT_DOSYASI, bekleyen_kayitlar)
            bekleyen_kayitlar = []

        if dogru_sayisi >= HEDEF_DOGRU and yanlis_sayisi >= HEDEF_YANLIS:
            break

    kayitlari_yaz(KAYIT_DOSYASI, bekleyen_kayitlar)

    son_kayitlar = kayitlari_oku(KAYIT_DOSYASI)
    print("Toplam kayit:", len(son_kayitlar))
    print("Dogru:", sum(r["dogru"] for r in son_kayitlar))
    print("Yanlis:", len(son_kayitlar) - sum(r["dogru"] for r in son_kayitlar))
    print(f"Sure: {(time.time() - basla) / 60:.1f} dk")


if __name__ == "__main__":
    main()
