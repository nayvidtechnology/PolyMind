import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from multimod.core.abi import validate_mm_abi


def test_v10_T_cap_supported_and_normalized():
    meta = {
        "version": "1.0",
        "projector": {
            "T_cap": 64,
            "d_model": 2048,
            "recipe_hint": "base",
            "encoder_trainable": "none",
        },
    }
    norm = validate_mm_abi(meta)
    assert norm["projector"]["t_cap"] == 64


def test_v11_t_cap_canonical():
    meta = {
        "version": "1.1",
        "projector": {
            "t_cap": 64,
            "d_model": 2048,
            "recipe_hint": "share",
            "encoder_trainable": "last_k_blocks",
        },
    }
    norm = validate_mm_abi(meta)
    assert norm["projector"]["t_cap"] == 64
