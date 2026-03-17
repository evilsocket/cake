#!/usr/bin/env python3
"""Compare two WAV files by computing waveform statistics and correlation."""
import sys
import struct
import math

def read_wav(path):
    """Read 16-bit mono WAV, return float samples."""
    with open(path, 'rb') as f:
        riff = f.read(4)
        assert riff == b'RIFF', f"Not RIFF: {riff}"
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b'WAVE'
        # Find data chunk
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'data':
                break
            f.read(chunk_size)
        data = f.read(chunk_size)
    samples = struct.unpack(f'<{len(data)//2}h', data)
    return [s / 32768.0 for s in samples]

def stats(samples):
    n = len(samples)
    if n == 0:
        return {"n": 0}
    mean = sum(samples) / n
    std = math.sqrt(sum((s - mean)**2 for s in samples) / n) if n > 1 else 0
    mx = max(samples)
    mn = min(samples)
    # RMS
    rms = math.sqrt(sum(s*s for s in samples) / n)
    # Zero crossing rate
    zc = sum(1 for i in range(1, n) if (samples[i] >= 0) != (samples[i-1] >= 0)) / n
    return {"n": n, "mean": mean, "std": std, "min": mn, "max": mx, "rms": rms, "zcr": zc}

def correlation(a, b):
    """Pearson correlation between two equal-length sample arrays."""
    n = min(len(a), len(b))
    if n < 2:
        return 0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da = math.sqrt(sum((a[i]-ma)**2 for i in range(n)))
    db = math.sqrt(sum((b[i]-mb)**2 for i in range(n)))
    if da < 1e-10 or db < 1e-10:
        return 0
    return num / (da * db)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} reference.wav test.wav")
        sys.exit(1)

    ref = read_wav(sys.argv[1])
    test = read_wav(sys.argv[2])

    rs = stats(ref)
    ts = stats(test)

    print(f"Reference: {rs['n']} samples, rms={rs['rms']:.4f}, zcr={rs['zcr']:.4f}, range=[{rs['min']:.4f},{rs['max']:.4f}]")
    print(f"Test:      {ts['n']} samples, rms={ts['rms']:.4f}, zcr={ts['zcr']:.4f}, range=[{ts['min']:.4f},{ts['max']:.4f}]")

    # Compare statistics
    rms_ratio = ts['rms'] / rs['rms'] if rs['rms'] > 0 else 0
    zcr_ratio = ts['zcr'] / rs['zcr'] if rs['zcr'] > 0 else 0

    print(f"\nRMS ratio: {rms_ratio:.2f} (should be ~1.0)")
    print(f"ZCR ratio: {zcr_ratio:.2f} (should be ~1.0)")

    if min(len(ref), len(test)) > 100:
        corr = correlation(ref, test)
        print(f"Correlation: {corr:.4f} (>0.3 = similar structure, >0.7 = good match)")

    # Verdict
    is_noise = ts['zcr'] > 0.4  # White noise has ZCR ~0.5
    is_speech = 0.05 < ts['zcr'] < 0.35 and ts['rms'] > 0.01
    ref_is_speech = 0.05 < rs['zcr'] < 0.35 and rs['rms'] > 0.01

    print(f"\nReference looks like: {'SPEECH' if ref_is_speech else 'NOISE/SILENCE'}")
    print(f"Test looks like:      {'SPEECH' if is_speech else 'NOISE' if is_noise else 'SILENCE/OTHER'}")

    if is_speech and ref_is_speech and rms_ratio > 0.3 and rms_ratio < 3.0:
        print("\n✅ MATCH: Test audio has speech-like characteristics similar to reference")
    else:
        print("\n❌ MISMATCH: Test audio does not match reference characteristics")
