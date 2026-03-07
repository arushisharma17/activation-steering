# HumanEval-Java Benchmark for Activation Steering

This directory integrates the **HumanEval-Java** benchmark with the **activation-steering** project.  
It allows evaluating **baseline** vs. **steered** code generation across multiple models and datasets.

---

## 📁 Directory Overview

```
human-eval-java/
├── meta/                     # Scripts for generation & evaluation
│   ├── scripts/
│   │   ├── gen_hejava.py     # Generation script
│   │   ├── eval_hejava.py    # pass@k evaluation
│   │   └── summarize_hejava_results.py
│   └── ...
│
├── hej_generations/          # Auto-generated results
│   ├── tssb/
│   │   ├── Qwen2.5-7B-Instruct/
│   │   │   ├── baseline/
│   │   │   └── steer-*/      # Each steering run
│   │   └── ...
│   └── manysstubs/
│       └── ...
│
├── run_hejava_all_models.sh  # Main SLURM script (multi-model, multi‑dataset)
└── README.md                 # This file
```

---

## 🚀 Running the Benchmark

### **Run all models (baseline + steered)**

```
sbatch run_hejava_all_models.sh
```

### **Run only specific models**

```
sbatch run_hejava_all_models.sh --models qwen7b qwen14b
```

### **Run baseline only**

```
sbatch run_hejava_all_models.sh --baseline
```

### **Run steered only**

```
sbatch run_hejava_all_models.sh --steered
```

### **Run evaluation only**

```
sbatch run_hejava_all_models.sh --eval-only
```

---

## 🧪 pass@k Evaluation Options

By default, the script evaluates:

- pass@1  
- pass@5  
- pass@10  

You can override:

```
--k-list 1,5
--max-k 5
```

Example:

```
sbatch run_hejava_all_models.sh --eval-only --k-list 1,10
```

---

## 📦 Dataset Selection

Steering vectors can come from different datasets:

- **TSSB (default)**  
- **ManySStuBs4J**  

Choose via:

```
--dataset tssb
--dataset manysstubs
```

Example:

```
sbatch run_hejava_all_models.sh --dataset manysstubs --steered
```

Each dataset gets its own output directory:

```
hej_generations/tssb/<MODEL_SLUG>/
hej_generations/manysstubs/<MODEL_SLUG>/
```

---

## 📊 Output Structure

Each model has:

```
hej_generations/<dataset>/<model_slug>/
│
├── baseline/
│   ├── samples.json
│   └── passk_results.json
│
└── steer-<vecID>-<timestamp>/
    ├── samples.json
    ├── metadata.json
    └── passk_results.json
```

You can easily compare baseline vs. steered runs.

---

## 📝 Summary Report

After all evaluations:

```
python meta/scripts/summarize_hejava_results.py
```

This script aggregates:

- pass@k across models
- baseline vs. steered comparison
- summary tables (CSV + JSON)

---

## 🙌 Notes

- Models are loaded from the local HF cache set in the SLURM script.
- Steering vectors must match the programming language of the benchmark (Java).
- You can extend the script to more models or datasets easily.

---



