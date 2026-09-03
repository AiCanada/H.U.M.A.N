# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""One-shot vendor of OpenFold (pl_upgrades) into rbase._ext.openfold."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".tmp_openfold" / "openfold"
DST = ROOT / "src" / "rbase" / "_ext" / "openfold"
PROPS_SRC = ROOT / "src" / "rbase" / "_patch" / "openfold" / "stereo_chemical_props.txt"
LICENSE_SRC = ROOT / ".tmp_openfold" / "LICENSE"

SKIP_DIR_NAMES = {"csrc", "__pycache__", "tests"}
IMPORT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<kind>from|import)\s+openfold(?P<rest>.*)$",
    re.M,
)

def rewrite_source(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        indent, kind, rest = match.group("indent", "kind", "rest")
        return f"{indent}{kind} rbase._ext.openfold{rest}"

    text = IMPORT_RE.sub(repl, text)
    text = text.replace(
        'resources.read_text("openfold.resources"',
        'resources.read_text("rbase._ext.openfold.resources"',
    )
    return text

def copy_tree() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    for path in SRC.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(SRC)
        dest = DST / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py":
            dest.write_text(rewrite_source(path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(path, dest)

def write_support_files() -> None:
    (DST / "resources").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROPS_SRC, DST / "resources" / "stereo_chemical_props.txt")
    shutil.copy2(LICENSE_SRC, DST / "LICENSE.txt")
    (DST / "README.md").write_text(
        "Vendored OpenFold Python package (Apache-2.0).\n\n"
        "Source: https://github.com/aqlaboratory/openfold/tree/pl_upgrades\n"
        "Used for residue constants, rigid/atom transforms, and optional\n"
        "sequence-representation generation. No `import openfold` from PyPI.\n",
        encoding="utf-8",
    )
    (DST / "__init__.py").write_text(
        '"""Vendored OpenFold (AlQuraishi Lab / DeepMind, Apache-2.0)."""\n'
        "from __future__ import annotations\n",
        encoding="utf-8",
    )

def patch_attention_import() -> None:
    prim = DST / "model" / "primitives.py"
    text = prim.read_text(encoding="utf-8")
    text = text.replace(
        "from rbase._ext.openfold.utils.kernel.attention_core import attention_core\n",
        "",
    )
    old = "            o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))"
    new = (
        "            from rbase._ext.openfold.utils.kernel.attention_core "
        "import attention_core\n"
        "            o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))"
    )
    if old not in text:
        raise SystemExit("Could not patch attention_core use in primitives.py")
    prim.write_text(text.replace(old, new), encoding="utf-8")

    core = DST / "utils" / "kernel" / "attention_core.py"
    text = core.read_text(encoding="utf-8")
    text = text.replace(
        'attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")',
        "try:\n"
        '    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\n'
        "except ModuleNotFoundError:\n"
        "    attn_core_inplace_cuda = None",
    )
    text = text.replace(
        "        attn_core_inplace_cuda.forward_(\n"
        "            attention_logits, \n"
        "            reduce(mul, attention_logits.shape[:-1]),\n"
        "            attention_logits.shape[-1],\n"
        "        )",
        "        if attn_core_inplace_cuda is not None:\n"
        "            attn_core_inplace_cuda.forward_(\n"
        "                attention_logits,\n"
        "                reduce(mul, attention_logits.shape[:-1]),\n"
        "                attention_logits.shape[-1],\n"
        "            )\n"
        "        else:\n"
        "            attention_logits.copy_(torch.softmax(attention_logits, dim=-1))",
    )
    text = text.replace(
        "        attn_core_inplace_cuda.backward_(\n"
        "            attention_logits,\n"
        "            grad_output.contiguous(),\n"
        "            v.contiguous(), # v is implicitly transposed in the kernel\n"
        "            reduce(mul, attention_logits.shape[:-1]),\n"
        "            attention_logits.shape[-1],\n"
        "            grad_output.shape[-1],\n"
        "        )",
        "        if attn_core_inplace_cuda is not None:\n"
        "            attn_core_inplace_cuda.backward_(\n"
        "                attention_logits,\n"
        "                grad_output.contiguous(),\n"
        "                v.contiguous(),\n"
        "                reduce(mul, attention_logits.shape[:-1]),\n"
        "                attention_logits.shape[-1],\n"
        "                grad_output.shape[-1],\n"
        "            )\n"
        "        else:\n"
        "            # Match the inplace kernel: attention_logits holds softmax weights.\n"
        "            attn = attention_logits\n"
        "            grad_attn = torch.matmul(grad_output, v.transpose(-1, -2))\n"
        "            grad_attn = attn * (\n"
        "                grad_attn - (grad_attn * attn).sum(dim=-1, keepdim=True)\n"
        "            )\n"
        "            attention_logits.copy_(grad_attn)",
    )
    core.write_text(text, encoding="utf-8")

    struct = DST / "model" / "structure_module.py"
    text = struct.read_text(encoding="utf-8")
    text = text.replace(
        'attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")',
        "try:\n"
        '    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\n'
        "except ModuleNotFoundError:\n"
        "    attn_core_inplace_cuda = None",
    )
    text = text.replace(
        "            # in-place softmax\n"
        "            attn_core_inplace_cuda.forward_(\n"
        "                a,\n"
        "                reduce(mul, a.shape[:-1]),\n"
        "                a.shape[-1],\n"
        "            )",
        "            # in-place softmax\n"
        "            if attn_core_inplace_cuda is not None:\n"
        "                attn_core_inplace_cuda.forward_(\n"
        "                    a,\n"
        "                    reduce(mul, a.shape[:-1]),\n"
        "                    a.shape[-1],\n"
        "                )\n"
        "            else:\n"
        "                a.copy_(torch.softmax(a, dim=-1))",
    )
    if "attn_core_inplace_cuda is not None" not in text:
        raise SystemExit("Could not patch attn_core_inplace_cuda use in structure_module.py")
    struct.write_text(text, encoding="utf-8")

def patch_script_utils() -> None:
    """Keep prep_output importable without pdbfixer / Lightning DeepSpeed."""
    path = DST / "utils" / "script_utils.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from rbase._ext.openfold.model.model import AlphaFold\n"
        "from rbase._ext.openfold.np import residue_constants, protein\n"
        "from rbase._ext.openfold.np.relax import relax\n"
        "from rbase._ext.openfold.utils.import_weights import (\n"
        "    import_jax_weights_,\n"
        "    import_openfold_weights_\n"
        ")\n"
        "\n"
        "from pytorch_lightning.utilities.deepspeed import (\n"
        "    convert_zero_checkpoint_to_fp32_state_dict\n"
        ")\n",
        "from rbase._ext.openfold.np import residue_constants, protein\n"
        "from rbase._ext.openfold.utils.import_weights import (\n"
        "    import_jax_weights_,\n"
        "    import_openfold_weights_\n"
        ")\n",
    )
    old_load = (
        "def load_models_from_command_line(config, model_device, openfold_checkpoint_path, jax_param_path, output_dir):\n"
        "    # Create the output directory\n"
    )
    new_load = (
        "def load_models_from_command_line(config, model_device, openfold_checkpoint_path, jax_param_path, output_dir):\n"
        "    from rbase._ext.openfold.model.model import AlphaFold\n"
        "    from pytorch_lightning.utilities.deepspeed import (\n"
        "        convert_zero_checkpoint_to_fp32_state_dict,\n"
        "    )\n"
        "\n"
        "    # Create the output directory\n"
    )
    if old_load not in text:
        raise SystemExit("Could not patch load_models_from_command_line in script_utils.py")
    text = text.replace(old_load, new_load)
    old_relax = (
        "def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output=False):\n"
        "    amber_relaxer = relax.AmberRelaxation(\n"
    )
    new_relax = (
        "def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output=False):\n"
        "    from rbase._ext.openfold.np.relax import relax\n"
        "\n"
        "    amber_relaxer = relax.AmberRelaxation(\n"
    )
    if old_relax not in text:
        raise SystemExit("Could not patch relax_protein in script_utils.py")
    path.write_text(text.replace(old_relax, new_relax), encoding="utf-8")

def rewrite_confrover_imports() -> None:
    skip_roots = {DST}
    files = []
    for path in (ROOT / "src" / "rbase").rglob("*.py"):
        if any(skip in path.parents or skip == path.parent for skip in skip_roots):
            if DST in path.parents or path.parent == DST:
                continue
        files.append(path)
    for path in (ROOT / "tests").rglob("*.py"):
        files.append(path)
    for path in files:
        text = path.read_text(encoding="utf-8")
        new = IMPORT_RE.sub(
            lambda m: f"{m.group('indent')}{m.group('kind')} rbase._ext.openfold{m.group('rest')}",
            text,
        )
        if new != text:
            path.write_text(new, encoding="utf-8")
            print(f"rewrote {path.relative_to(ROOT)}")

if __name__ == "__main__":
    if not SRC.is_dir():
        raise SystemExit(f"Missing clone at {SRC}")
    copy_tree()
    write_support_files()
    patch_attention_import()
    patch_script_utils()
    rewrite_confrover_imports()
    print(f"Vendored OpenFold to {DST}")
