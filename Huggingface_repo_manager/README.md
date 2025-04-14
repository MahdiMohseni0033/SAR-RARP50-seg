# 🤗 Huggingface Repository Manager

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.6%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

*A sleek command-line utility for effortless Huggingface repository management*

</div>

## ✨ Features

- **📤 Upload Files** — Transfer local files to your Huggingface repositories
- **🗑️ Delete Files** — Remove files with built-in confirmation safeguards
- **📊 Repository Info** — Access detailed metadata about any repository
- **🔍 File Browser** — Explore repository content with tree structure view

## 🚀 Usage

```bash
python hf_manager.py
```

<details>
<summary>Interactive Workflow</summary>

1. Enter a repository ID (e.g., 'username/model-name')
2. Provide your Huggingface API token (or use environment variables)
3. Select repository type (model, dataset, or space)
4. Choose operations from the interactive menu

</details>

## 🔐 Authentication

| Method | Description |
|--------|-------------|
| Direct input | Enter token at runtime prompt |
| Environment | Set `HF_TOKEN` environment variable |
| CLI cache | Use existing `huggingface-cli login` |

## 🧩 Project Integration

```
Project Structure
├── config.yaml
├── convert_to_yolo.py
├── custom_segmentation.yaml
├── dataset_manager.py
├── Huggingface_repo_manager    ← You are here
│   └── hf_manager.py
├── inference.py
├── inf_utils.py
...
```

This utility handles repository interactions while other components manage dataset preparation, training configuration, and model inference.

## 📋 Requirements

Requires the `huggingface_hub` package, included in the project's requirements.txt.

