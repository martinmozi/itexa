# 04 — Veľké jazykové modely

Od architektúry transformera po praktickú prácu s dnešnými LLM.

| Dokument | O čom je |
|---|---|
| [01-transformer-siete.md](01-transformer-siete.md) | self-attention (Q, K, V), multi-head, positional encoding, autoregresívne generovanie a dekódovanie |
| [02-transformer-vnutro.md](02-transformer-vnutro.md) | **A:** tokenizácia a tokenizéry, text → vektory, okno vs. šírka, čo model kvôli tokenom nevidí · **B:** rozmery a parametre, ako model rastie (šírka vs. hĺbka a paralelizácia), reziduálny prúd, feed-forward a MoE · **C:** vznik výstupného tokenu a slova, KV cache, krátka vs. dlhá správa, limity kontextu, in-context learning, presnosť a kvantizácia |
| [03-llm-trening.md](03-llm-trening.md) | dáta → pretraining → base model → SFT/instruction tuning → Instruct model → preferenčné ladenie (RLHF, DPO, RLVR); teacher forcing a maskovanie loss, iterácie generovania krok po kroku, perplexita a benchmarky, scaling laws a Chinchilla, tréning na tisíckach GPU |
| [04-llm-modely.md](04-llm-modely.md) | proprietárne vs. open-weight vs. open-source, výber modelu podľa úlohy, právne a etické mantinely |
| [05-embeddings.md](05-embeddings.md) | ako sa z textu stane vektor: tokenizácia, token embeddingy, transformer vrstvy, pooling, normalizácia |
| [06-rag.md](06-rag.md) | chunking, indexovanie, vyhľadávanie, reranking, výpočtové nároky, pokročilý retrieval |
| [07-fine-tuning-lora.md](07-fine-tuning-lora.md) | LoRA, QLoRA, kedy fine-tuning áno a kedy radšej RAG |

← [03 — Tréning modelov](../03-trening-modelov/README.md) · ďalej [05 — Praktické](../05-prakticke/README.md) →
