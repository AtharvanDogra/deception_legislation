# Language Models can Subtly Deceive Without Lying (ACL'25): Code and Data

<img width="1786" height="741" alt="main" src="https://github.com/user-attachments/assets/0e7d15c7-663c-4b4f-ade0-4c6a5e0a12dc" />

This repository hosts code and data utilities for the ACL 2025 long paper “Language Models can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing in Legislation,” including scripts to reproduce the Lobbyist–Critic simulation, re-planning/resampling optimization, evaluation pipelines, and the LobbyLens dataset.

### Quick links
- Paper (ACL Anthology): https://aclanthology.org/2025.acl-long.1600/
- Dataset (Hugging Face): https://huggingface.co/datasets/atharvan/LobbyLens

### Citation
Please cite the ACL 2025 paper if this code or dataset is used in research.
```
@inproceedings{dogra-etal-2025-language,
    title = "Language Models can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing in Legislation",
    author = "Dogra, Atharvan  and
      Pillutla, Krishna  and
      Deshpande, Ameet  and
      Sai, Ananya B.  and
      Nay, John J  and
      Rajpurohit, Tanmay  and
      Kalyan, Ashwin  and
      Ravindran, Balaraman",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1600/",
    doi = "10.18653/v1/2025.acl-long.1600",
    pages = "33367--33390",
    ISBN = "979-8-89176-251-0",
    abstract = "We explore the ability of large language models (LLMs) to engage in subtle deception through strategically phrasing and intentionally manipulating information. This harmful behavior can be hard to detect, unlike blatant lying or unintentional hallucination. We build a simple testbed mimicking a legislative environment where a corporate \textit{lobbyist} module is proposing amendments to bills that benefit a specific company while evading identification of this benefactor. We use real-world legislative bills matched with potentially affected companies to ground these interactions. Our results show that LLM lobbyists can draft subtle phrasing to avoid such identification by strong LLM-based detectors. Further optimization of the phrasing using LLM-based re-planning and re-sampling increases deception rates by up to 40 percentage points.Our human evaluations to verify the quality of deceptive generations and their retention of self-serving intent show significant coherence with our automated metrics and also help in identifying certain strategies of deceptive phrasing.This study highlights the risk of LLMs' capabilities for strategic phrasing through seemingly neutral language to attain self-serving goals. This calls for future research to uncover and protect against such subtle deception."
}
```

## Overview

This project studies subtle deception by LLMs via strategic phrasing that preserves technical truth while obscuring self-serving intent in a legislative testbed.  A “Lobbyist” agent drafts amendments to real US congressional bills to covertly benefit a specific company, while a “Critic” agent tries to identify the hidden benefactor from candidate companies matched to the bill.  The method includes iterative re-planning and re-sampling, which increases deception success (reducing correct identifications by up to ~40 percentage points across models) while largely retaining intended benefits.

<img width="8000" height="1800" alt="parameters" src="https://github.com/user-attachments/assets/feec4f43-4ce5-4f24-b09a-59e91359c425" />


### Artifact contributions
- LobbyLens: a grounded dataset linking 936 bills to 4–12 relevant public companies each (4,439 bill–company pairs) with metadata.
- A simple, reproducible Lobbyist–Critic framework with pairwise company comparisons and Bradley–Terry–Luce spectral ranking for identification.

## Repository structure

- memory: Sample simulation outputs from Qwen 72B Model. Also, use as a reference for results_notebook.ipynb for main results calculation.
- models: contains basic agent classes, some model classes for llama.cpp models, and other utilities.
- utility_capture: contains results, code to calculate, and notebook to collate results.
- results_notebook.ipynb: sample results calcualtion
- simulator.py: main script to run simulations
- stronger_critic.py: simpler critic evaluation script. We used it to evaluate previous simulations against stronger critic.

## Models Used
Supports open-source llama.cpp version of chat LLMs used in the paper (e.g., Mistral 7B Instruct, Mixtral 8×7B, Yi 34B, Qwen 7B/14B/72B), plus API-backed models if configured.


## LobbyLens Dataset

LobbyLens links US congressional bills (108th–118th) with potentially affected public companies via embedding-based similarity (BGE-Large-en/FlagEmbedding), filtering to ensure ≥4 candidates per bill and summaries ≤600 tokens for context feasibility.  Final set: 936 bills × 4–12 companies = 4,439 bill–company pairs, with metadata fields such as title, congress, bill type, summary, policy area, state, company name/ticker, and 10-K business description.[1]

- Component Attribution: Bills from govinfo.gov (CC0-1.0) via prior HF datasets; company descriptions from SEC 10-K with references as in paper.

### Dataset fields

Core fields per example include title, congress, bill_type, bill_text/summary, policy_area, state, company_name, company_ticker, and business_description; see Table 1 in the paper for descriptions.


## Safety and ethical notes

This repository demonstrates and measures subtle deception capabilities of LLMs for scientific study and red-teaming; it does not introduce novel adversarial training and uses off-the-shelf LLMs.  Exercise caution when applying or extending this code to high-stakes domains and ensure alignment with legal/ethical guidelines.[1]

## Known limitations

- Focuses on LLM-vs-LLM deception; human-facing deception evaluation is limited.
- Legislative process abstractions; real-world trade-offs are more complex.
- Strongest critics partly API-based due to compute/cost; broader frontier-model coverage is future work.


## License

- Code: GNU General Public License v3.0,
- Data: follows original sources’ terms; bill summaries CC0-1.0 via US government sources; company descriptions from SEC filings; see paper for details.

## Acknowledgments

Partially supported by compute credits from OpenAI, detailed in the paper; thanks to human evaluators and prior datasets used to build LobbyLens. This code uses the open-source [guidance framework](https://github.com/guidance-ai/guidance) for their simpler inference structures.
