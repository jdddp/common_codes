import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOv8-nano style ONNX to RKNN.")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--dataset", type=str, required=True, help="Calibration image list txt.")
    parser.add_argument("--quant", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    ret = rknn.config(
        target_platform=args.target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quant_img_RGB2BGR=False,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    ret = rknn.build(do_quantization=args.quant, dataset=args.dataset if args.quant else None)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")

    rknn.release()
    print(f"exported rknn to {output_path}")


if __name__ == "__main__":
    main()
