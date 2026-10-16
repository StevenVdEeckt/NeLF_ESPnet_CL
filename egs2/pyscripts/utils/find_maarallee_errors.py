import soundfile as sf
import sys

wav_scp = "/esat/audioslave/svandere/espnet/asr/ml/dump/raw/Maarallee/wav.scp"

bad = []
with open(wav_scp, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        utt, path = line.split(maxsplit=1)

        # Kaldi pipes end with |; soundfile can't open that directly.
        if path.endswith("|"):
            bad.append((utt, "PIPE_IN_WAV_SCP", path))
            continue

        try:
            info = sf.info(path)
            # sanity checks (tweak thresholds as you like)
            if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                bad.append((utt, "BAD_INFO", str(info)))
            if info.duration > 3600:  # > 1 hour is suspicious for your 30s segments
                bad.append((utt, f"TOO_LONG_{info.duration:.1f}s", path))
        except Exception as e:
            bad.append((utt, f"READ_FAIL: {e}", path))

print("Bad entries:", len(bad))
for b in bad[:50]:
    print(*b, sep="\t")
if len(bad) > 50:
    print("... (truncated)")
