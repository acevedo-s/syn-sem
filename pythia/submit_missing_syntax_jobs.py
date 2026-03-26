import subprocess
from pathlib import Path


REPO_ROOT = Path("/home/acevedo/syn-sem")
SEM_ROOT = REPO_ROOT / "datasets" / "activations" / "sem" / "second"
SYN_SECOND_ROOT = REPO_ROOT / "datasets" / "activations" / "syn" / "second"
SYN_THIRD_ROOT = REPO_ROOT / "datasets" / "activations" / "syn" / "third"
HF_ROOT = REPO_ROOT / "hf_extract_activations"
LANGUAGES = ["arabic", "chinese", "english", "german", "italian", "spanish", "turkish"]


def has_full_semantics(model_name: str) -> bool:
    base = SEM_ROOT / model_name / "matching"
    return all((base / language).is_dir() for language in LANGUAGES)


def has_syntax(model_name: str) -> bool:
    path_A = SYN_SECOND_ROOT / model_name / "matching" / "english" / "0"
    path_B = SYN_THIRD_ROOT / model_name / "matching" / "english" / "1"
    return path_A.is_dir() and path_B.is_dir()


def missing_models():
    for model_dir in sorted(SEM_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if has_full_semantics(model_name) and not has_syntax(model_name):
            yield model_name


def submit(model_name: str) -> str:
    cmd = ["sbatch", "sactivations.sh", "english", "syn", "matching", model_name]
    out = subprocess.check_output(cmd, cwd=HF_ROOT, text=True)
    return out.strip()


def main():
    models = list(missing_models())
    if not models:
        print("No missing syntax jobs found.")
        return

    for model_name in models:
        result = submit(model_name)
        print(f"{model_name}: {result}")


if __name__ == "__main__":
    main()
