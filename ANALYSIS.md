# English–Hindi machine translation: measured analysis

## Experimental setup and scope

The IIT Bombay English–Hindi Parallel Corpus was loaded from [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi). There are 5,000 cleaned training pairs, 200 validation pairs and 200 test pairs. The test and validation rows came from the provided official splits; training was shuffled with seed 4403. NFC normalization, whitespace cleanup, a 600-character limit and source-level deduplication were applied. All four CSV files have the same 200 IDs, sources and references in the same order. The test subset consists of the first 200 eligible official test items, many drawn from a few contiguous news stories; it is a narrow sample.

The scratch Transformer uses 3 encoder and 3 decoder layers with a training-only joint SentencePiece vocabulary. mT5-small receives a full fine-tune. Llama 3.2 1B Instruct and Gemma 3 270M IT receive rank-8 LoRA adapters on query and value projections. The main evaluation is greedy decoding with at most 96 new tokens per sentence; the temperature experiment is separate. All runs used a Tesla T4, seed 4403, batch size 4 and gradient accumulation 4. Transformer/Llama/Gemma used 300 optimizer steps; mT5 used 900 steps and learning rate 0.0003 after an initial 300-step run produced pretraining sentinel tokens. The final 900-step run is the one reported here. Each JSON records exact hyperparameters and hardware.

## Quantitative comparison

| Model | BLEU | chrF | Train min | Inference s/sentence | Total params M | Trainable params M | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch Transformer | 0.08 | 5.09 | 0.62 | 0.46 | 7.58 | 7.58 | 300 |
| mT5-small | 0.50 | 9.46 | 11.85 | 0.84 | 300.18 | 300.18 | 900 |
| Llama 3.2 1B Instruct | **8.87** | **33.89** | 30.36 | 1.54 | 1235.81 | 0.85 | 300 |
| Gemma 3 270M IT | 6.25 | 25.92 | 5.78 | 1.13 | 268.10 | 0.37 | 300 |

BLEU and chrF are corpus-level SacreBLEU scores, both higher-is-better. Reported training time covers optimization, excludes download and validation; inference includes tokenization and one-example generation with CUDA synchronization, excludes model loading. The reported parameter counts are exact `numel()` counts for the loaded model and trainable weights. LoRA updates only about 0.85 million parameters in Llama and 0.37 million in Gemma; the full model still has to be loaded for inference. The [score plot](results/scores.png) is generated from the measured JSON files.

Llama leads on both quality measures but takes the longest to train and infer. Gemma is second in quality and cheaper on this GPU. mT5 and the from-scratch Transformer are very weak in these runs. This is **not** a controlled test of architecture alone: model size, pretraining objective, instruction tuning, tokenizer, adaptation method and training-step count all differ. mT5 was trained for 900 steps while the others received 300; its extra training did not make it competitive. The small scratch model had far less data and training than typical production MT systems. Results on 200 news-heavy examples cannot establish performance for other domains or dialects.

## Translation inspection

The full unabridged source, reference and prediction for every ID are in `results/<model>_translations.csv`. The examples below are observations from those files, not manually corrected outputs.

| Test ID and source | Reference and observed outputs | What it shows |
| --- | --- | --- |
| **0:** “A black box in your car?” | Reference: “आपकी कार में ब्लैक बॉक्स?” Llama: “आपके कार में क्या पुराना ब्लैक बॉक्स है?” Gemma: “एक काला बॉक्स आपके कार में है?” mT5: “अखंड वाला नंबर वाला नंबर?” Transformer: “⁇” | Llama and Gemma convey a box in a car but add or alter wording; Llama invents “old.” mT5 and Transformer fail to preserve the question. |
| **5:** “The tea party is aghast.” | Reference: “चाय पार्टी भौचक्की है।” Llama: “ती पार्टी को दुख हुआ है।” Gemma: “दही पार्टी में अफ़्दा है।” mT5: “इस सभा में यह बात है।” | “Tea party” is culturally and politically ambiguous. Gemma's “दही पार्टी” changes tea to yogurt; none preserves the reference's sense cleanly. The source context should be reviewed before settling on a cultural rendering. |
| **17:** “The federal tax itself, 18.4 cents per gallon, hasn't gone up in 20 years.” | Llama: “फेडरल टैक्स के हिसाब से 18.4 सेंट की दर में 20 वर्षों में कोई बदलाव नहीं हुआ है।” Gemma: “यह संघीय कर, 18.4 cents प्रति gallon में नहीं बढ़ गया है।” mT5 omits the numbers. | Llama retains both numeric facts; Gemma omits 20 years and mixes English units; mT5 loses critical details. |
| **68:** “… on August 26th the police recovered the body of Jahid Iqbal from the Jhelum river in Padshahi Park.” | Llama: “… 26 अगस्त को पादशाही पार्क में जेलमूल नदी के पास जाहिद iqbal का शरीर पुलिस ने प्राप्त कर लिया था।” Gemma: “… 26 अगस्त को … जहिशि … जल्मी नदी … एक शव …” | Both distort proper names; Gemma drops “Iqbal.” Llama retains the date and name partially but corrupts the river name. A human must verify names and locations. |
| **139:** Source reports a *plea* about media portrayal of Asaram and a court refusal to ban coverage. | Llama output ends midword after a changed account of the Supreme Court; Gemma says “… सुप्रीम कोर्ट ने उसे इस बात पर कोई भी रोक नहीं दी।” mT5 repeats generic wording. | In a legal and reputationally sensitive passage, omissions and changed attribution could mislead readers. The source is a reported allegation, not an established assertion by this report. |
| **179:** Source says people involved in bombings **intended** to cause a stampede during a rally. | Llama: “भारतीय मुजाहिदीनों ने सड़क पर प्रदर्शन के दौरान मारपीट की योजना बनाई थी…” Gemma: “भारतीय उद्रदियों के साथ उग्रों की सभा में शामिल लोगों को एक बड़ा संकट का सामना करना पड़ा…” | Both change actors or the planned act; the distinction between intent and an event is crucial in sensitive reporting. |
| **198:** Source gives cricket facts: Virat turns 25 next week, 17 one-day centuries, 16 linked to wins. | Llama retains 25, 17 and 16 but uses future tense for the scored centuries; Gemma confuses ages, days and counts. | Entity and number retention alone cannot guarantee the event timing is translated correctly. |

Repeated strings are also common in the two encoder–decoder outputs. For example, mT5 on ID 1 repeats “नई” many times, and the scratch Transformer on ID 1 cycles through “क के लिए” and “क में”. These are observed failure modes under the chosen budget, not evidence that either architecture always behaves this way. Even the strongest model makes material errors and should not be treated as a reliable translator for consequential content.

## Sampling temperature analysis

The `temperature.json` files sample the same first ten test IDs at temperatures 0.3, 0.7 and 1.0 with a fixed random seed. They are **separate exploratory generations**, not the greedy outputs used for BLEU/chrF. At ID 0, Llama's 0.3 sample says “आपकी कार में एक काली बॉक्स है?”, while its 1.0 sample says “तुमके वाहन में एक रेनबो है?”; Gemma's 1.0 sample asks whether there is a cobra in the car. The rainbow and cobra are not in the English source. At ID 5, Gemma's 0.3 sample remains about a party, while its 0.7 sample becomes largely unintelligible. These examples show greater variability and potential invented details at higher temperature **in this small sample**. One sample per setting on ten items does not support a reliable numeric temperature-quality ranking. Greedy or low-temperature decoding is safer for factual translation, but still requires review.

## Responsible use and mitigation

Observed risks include fabricated or lost entities (IDs 0 and 68), incorrect numbers or timing (IDs 17 and 198), cultural mistranslation (ID 5), and changes to sensitive police or legal claims (IDs 139 and 179). The dataset contains names and potentially sensitive news; preserve privacy and do not infer guilt from machine output. No broad demographic bias conclusion can be drawn from this small selected set, but biased gender, religion, caste or dialect renderings remain a concern requiring targeted bilingual evaluation.

For safer deployment: check names, numbers, dates and negation against the source; use bilingual human review for legal, medical, law-enforcement and other consequential text; build test sets across dialects, communities and domains; retain source text alongside translations; flag low-confidence or repetitive outputs; allow corrections and incident reports; and restrict access to sensitive inputs/outputs. Evaluate references too, because parallel corpora may contain ambiguous or imperfect translations (for example, the location phrasing in ID 68). Avoid presenting these outputs as authoritative facts.

## Conclusion

**Llama 3.2 1B Instruct is the best-performing model in this experiment** on both BLEU (8.87) and chrF (33.89) and usually preserves more of the source than the other models in inspected examples. Its 30.36-minute optimization and 1.54-second per-sentence inference are the highest costs here. Gemma offers a smaller, faster decoder-only alternative with lower but substantially better scores than the two encoder–decoder runs. All four have important errors; the ranking is conditional on this corpus subset, hardware, training budgets and decoding settings. A larger, more varied test set and better-matched training budgets would be needed for a general architectural conclusion.
