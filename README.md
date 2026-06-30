# Cosmos-T1 Top1-Top2 Confidence Prediction

Bu repo, Hesaplamalı Anlambilim proje konusu 4 kapsamında hazırlanan
top1-top2 token olasılık farkı analizini içerir.

## Amaç

Cosmos-T1 modelinin Türkçe matematik sorularına verdiği cevapların doğru mu
yanlış mı olduğunu, cevap üretimi sırasında oluşan top1-top2 token olasılık
farklarından tahmin etmek.

Her token üretim adımında:

```text
top1-top2 farkı = P(en olası token) - P(ikinci en olası token)
```

farkı kaydedildi. Daha sonra her soru için bu fark dizisinden istatistiksel
özellikler çıkarıldı ve klasik makine öğrenmesi modelleri eğitildi.

## Kullanılan Model ve Veri Kümesi

- Model: [ytu-ce-cosmos/Turkish-Gemma-9b-T1](https://huggingface.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1)
- Veri kümesi: [ytu-ce-cosmos/gsm8k_tr](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr)

## Deney Özeti

Toplanan kayıtlar:

- Toplam kayıt: 357
- Doğru cevap: 187
- Yanlış cevap: 170

Dengeli eğitim/test ayrımı:

- Eğitim: 120 doğru + 120 yanlış = 240
- Test: 50 doğru + 50 yanlış = 100

## Model Sonuçları

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.720 | 0.844 | 0.540 | 0.659 | 0.738 |
| Random Forest | 0.700 | 0.727 | 0.640 | 0.681 | 0.747 |
| Gradient Boosting | 0.720 | 0.720 | 0.720 | 0.720 | 0.762 |

En dengeli sonuç Gradient Boosting modeli ile elde edilmiştir.

## Dosya Yapısı

```text
notebooks/
  top1_top2_cosmos_t1.ipynb

reports/
  rapor_yeni.pdf
  rapor_yeni_duzeltilmis.tex

sunum/
  sunum dosyalari

figures/
  fig_token_level_diff.png
  fig_mean_diff_distribution.png
  fig_roc.png
  fig_confusion_matrix.png
  fig_feature_importance.png

data/
  cosmos_t1_kayitlar_768.jsonl

literature/
  ilgili literatur PDF dosyalari
```

## Not

Cosmos-T1 modeli uzun chain-of-thought çıktıları ürettiği için veri toplama
süreci GPU üzerinde zaman almıştır. Bu nedenle eğitim kümesi proje hedefindeki
1000 örneğe ulaşamamış; ancak test kümesi 50 doğru + 50 yanlış olacak şekilde
100 dengeli örnekten oluşturulmuştur.
