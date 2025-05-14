#!/bin/bash

set -e

python scripts/pii_rule_comparison.py \
  --docs-folder /home/ubuntu/s2pdf_dedupe_minhash_v1_with_no_pii/documents \
  --ref-rule "ft_lang_id_en_doc_v2__ft_lang_id_en_doc_v2__en:avg>0.5" \
  --hyp-rule "ft_lang_id_en_doc_v2__ft_lang_id_en_doc_v2__en:avg>0.4" \
  --output-dir results/pii_detection \


tinyhost results/pii_detection/*