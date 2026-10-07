"""
web_runtime_check.py -- run an exported model in the WEBSITE's runtime (onnxruntime-web 1.22.0,
WASM) under node, on real crops and simulated drawings, and compare with Python onnxruntime.
Catches any ONNX operation the browser cannot run before we find out in class.

    python scripts/web_runtime_check.py artifacts/candidate/cnn.onnx
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parent))
from contract import file_to_contract          # noqa: E402
from dataset import load_rows, MANIFEST        # noqa: E402
from canvas_sim import batch                   # noqa: E402

RUNNER = Path("/private/tmp/claude-501/-Users-thoffman/fffe024e-d91f-4ad7-8b04-58de29c6dfa2/scratchpad/ortweb")


def main(model):
    xs = [file_to_contract(MANIFEST.parent / r["file"]) for r in load_rows("val")[:10]]
    X, _ = batch(5, (40, 220), seed=7); xs += list(X)
    xs.append(np.full((1, 64, 64), 1.0, np.float32))                        # blank white canvas
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([x.ravel().tolist() for x in xs], f)
    out = subprocess.run(["node", "check.mjs", str(Path(model).resolve()), f.name], cwd=RUNNER, capture_output=True, text=True)
    if out.returncode:
        print("WEB RUNTIME FAILED:\n", out.stderr[-1500:]); sys.exit(1)
    web = np.array(json.loads(out.stdout))
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    py = np.array([s.run(["logits"], {"image": x[None]})[0][0] for x in xs])
    print(f"{model}: web runtime ran {len(xs)} inputs; max |web - python| logit diff {np.abs(web - py).max():.2e}; "
          f"same predicted class on all: {bool((web.argmax(1) == py.argmax(1)).all())}")


if __name__ == "__main__":
    main(sys.argv[1])
