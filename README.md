# Kinyarwanda POS Tagging: General vs Afro-centric Models

Part-of-speech tagger for Kinyarwanda trained on [MasakhaPOS](https://github.com/masakhane-io/masakhane-pos)
(Universal Dependencies, 17 tags), comparing **general multilingual** pretrained models with
**Afro-centric** ones, plus a local web UI to switch between the trained models.

**Question:** does an Afro-centric model give higher accuracy than a strong general model?

<!-- RESULTS_SUMMARY -->

## Models compared

The study uses two pairs. Each pair puts a general multilingual model against an Afro-centric model of the same size.

| Pair | General model | Afro-centric model | Why this pair |
|---|---|---|---|
| **Large (main)** | [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large), 560M params, 100 languages, no Kinyarwanda | [AfroXLMR-large-76L](https://huggingface.co/Davlan/afro-xlmr-large-76L), 560M, XLM-R large further pretrained on 76 African languages (incl. Kinyarwanda) | Same architecture and tokenizer, so the only difference is the African-language pretraining. This is the cleanest test of the question. |
| **Base (reference)** | [mBERT cased](https://huggingface.co/google-bert/bert-base-multilingual-cased), 178M, 104 Wikipedia languages, no Kinyarwanda | [AfriBERTa-large](https://huggingface.co/castorini/afriberta_large), 126M, trained from scratch on 11 African languages (incl. Kinyarwanda) | The classic BERT vs AfriBERTa choice, kept as a smaller-model reference point. |

Plain English BERT was not used, because it has never seen Kinyarwanda or any related language. mBERT is the multilingual BERT,
and it is the fair "BERT" baseline here.

## Training recipes

Every model is trained with the same two recipes:

1. **Plain fine-tune**: the MasakhaPOS paper protocol. Encoder plus a linear tag layer, trained on Kinyarwanda train only
   (757 sentences): 20 epochs, batch 32, learning rate 5e-5. The best dev epoch is kept.
2. **Full recipe**: built for this low-resource setting.
   - **Stage 1 (multilingual):** trained on all 20 MasakhaPOS languages, with Bantu languages weighted up.
     Kinyarwanda contributes its train split only.
   - **Stage 2 (Kinyarwanda):** fine-tuned on Kinyarwanda with:
     - a character CNN and prefix/suffix features, which help with Kinyarwanda's rich morphology and unseen words;
     - a CRF output layer.

Each (model, recipe) is trained with 5 random seeds. Dev picks everything (epochs, which checkpoint to ship),
and test is only reported, never used for choices.

## Evaluation

- **Test set:** MasakhaPOS Kinyarwanda test, 604 sentences and about 17k tokens.
  About 21% of test tokens never appear in training (**OOV**), so OOV accuracy is reported separately.
- **Accuracy:** mean ± standard deviation over seeds.
- **Pair comparisons:**
  - the accuracy gap (Afro-centric minus general);
  - a 95% confidence interval from a paired bootstrap over test sentences (10,000 resamples);
  - an exact permutation test over seeds;
  - counts of tokens only one model of the pair gets right.
- **Script:** `src/compare_pairs.py` writes `results/summary.json`, `results/pairs.json`,
  `results/disagreements.json` and `results/REPORT.md`.

<!-- RESULTS_TABLES -->

## Web UI (switch models, compare a pair, browse errors)

```bash
pip install -r requirements.txt
# put checkpoints in models/<recipe>_<model>/model.pt (see "Trained model" below), then:
python app/server.py            # open http://127.0.0.1:8790
```

- **Tag text:** pick any trained model, paste Kinyarwanda, see the tag of every word. Tick
  *Compare with its pair partner* to run both models of the pair and outline every word they tag differently.
- **Results:** the verdict for each pair (gap, confidence interval, p-value), the leaderboard, and per-tag F1 gaps.
- **Test set:** every test sentence where the two models of a pair disagree, with gold tags, and wrong tags shown in red.

Models run on your CPU and load on demand. Only one stays in memory unless you pass `--max_loaded 2`.

<!-- MODEL_DOWNLOAD -->

## Reproduce

**Google Colab (free T4 GPU):** upload this repo as `kinpos_colab.tar.gz` (folders `src colab slurm`), mount Drive, then:

```bash
tar xzf kinpos_colab.tar.gz -C /content/kinpos && bash /content/kinpos/colab/setup.sh
python /content/kinpos/colab/run_queue.py --code /content/kinpos --data /content/masakhane-pos/data \
       --out /content/drive/MyDrive/kinpos_pairs      # resumable; progress: python colab/status.py
```

The queue runs all 44 jobs in about 4 hours on a T4:
- plain fine-tuning: 4 models × 5 seeds;
- Stage 1: 4 runs;
- full recipe: 4 models × 5 seeds.

Results land in Drive. Then run:

```bash
python src/compare_pairs.py --runs <drive>/kinpos_pairs/runs --data_dir masakhane-pos/data --out results
```

**Slurm cluster (Princeton Adroit):** `slurm/` holds the same jobs as array scripts
(`cfg_pairs.txt`, `cfg_pairs_full.txt`, `stage1.slurm`, `ablate.slurm`). Run `download_backbones.sh` on a login node first.

## Repository layout

```
src/train_tagger.py      training / evaluation (encoder + optional char CNN, affix features, CRF)
src/kinpos/              data reading, batching, features, CRF, metrics, MasakhaPOS-style pre-tokenizer
src/compare_pairs.py     pair statistics + REPORT.md
src/tag_text.py          tag raw text from the command line
src/package_model.py     bundle a checkpoint + inference code as a zip
app/                     web UI (server.py + static/)
colab/                   resumable job queue for a single Colab GPU
slurm/                   the same experiments as Slurm job arrays
results/                 metrics, pair statistics, disagreements (generated)
```

## Credits and license notes

- **Data:** MasakhaPOS (Dione et al., 2023), CC BY-NC 4.0, so this work is for non-commercial use.
- **Pretrained models:** from their authors on Hugging Face (links above). Check each model card for its license.
