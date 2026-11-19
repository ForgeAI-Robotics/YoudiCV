"""Record a bounded, unsegmented DOV sample; never import robot controls."""
from pathlib import Path
import argparse
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60")
    root = Path(__file__).resolve().parents[1] / "voice_diagnostics"
    root.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S_"), dir=root))
    source = "alsa_input.usb-CF-IC_DOV_2025-0711-1800-00.analog-stereo"
    path = folder / "original_48k_stereo.wav"
    input("准备完成。等待开始信号。\n")
    process = subprocess.Popen([
        "parec", "--device=" + source, "--file-format=wav",
        "--rate=48000", "--channels=2", str(path),
    ])
    print(f"录音开始，窗口{args.seconds}秒。请说：送外卖到十五楼。", flush=True)
    try:
        process.wait(timeout=args.seconds)
        raise RuntimeError("录音进程提前结束")
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=3)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    print("原始录音已保留：" + str(path), flush=True)
    subprocess.run(["sox", str(path), "-r", "16000", "-c", "1",
                    str(folder / "mono_16k.wav")], check=True)


if __name__ == "__main__":
    main()
