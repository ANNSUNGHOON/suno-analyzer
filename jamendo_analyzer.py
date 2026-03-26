"""
Jamendo → Suno Analyzer Mode B 자동화 스크립트
Jamendo에서 CC 라이선스 음원을 가져와 Railway 백엔드로 분석
"""
import requests
import time
import json
import sys
import os

# ── Config ──
JAMENDO_CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID", "YOUR_CLIENT_ID")
ANALYZER_URL = os.getenv("ANALYZER_URL", "https://web-production-53cf6.up.railway.app")
JAMENDO_API = "https://api.jamendo.com/v3.0"

# ── Jamendo에서 트랙 검색 ──
def search_tracks(tags=None, genre=None, limit=10, offset=0, order="popularity_total"):
    params = {
        "client_id": JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "order": order,
        "include": "musicinfo",
        "audioformat": "mp32",
    }
    if tags:
        params["tags"] = tags
    if genre:
        params["fuzzytags"] = genre

    r = requests.get(f"{JAMENDO_API}/tracks/", params=params)
    data = r.json()

    if data.get("headers", {}).get("status") != "success":
        print(f"[ERROR] Jamendo API: {data.get('headers', {}).get('error_message', 'unknown')}")
        return []

    tracks = []
    for t in data.get("results", []):
        tracks.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artist_name"],
            "duration": t["duration"],
            "audio_url": t.get("audio"),
            "audiodownload_url": t.get("audiodownload"),
            "tags": t.get("musicinfo", {}).get("tags", {}),
            "genre": t.get("musicinfo", {}).get("tags", {}).get("genres", []),
            "license": t.get("license_ccurl", ""),
        })
    return tracks


# ── 오디오 다운로드 ──
def download_audio(url, filename="temp_audio.mp3"):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"[ERROR] Download failed: {r.status_code}")
        return None

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return filename


# ── Mode B 분석 요청 ──
def analyze_audio(filepath, label=""):
    with open(filepath, "rb") as f:
        files = {"file": (os.path.basename(filepath), f, "audio/mpeg")}
        data = {"ip": "jamendo-bot", "label": label}
        r = requests.post(f"{ANALYZER_URL}/analyze-audio", files=files, data=data, timeout=300)

    if r.status_code != 200:
        print(f"[ERROR] Analysis failed: {r.status_code} - {r.text[:200]}")
        return None

    return r.json()


# ── 메인 실행 ──
def run_batch(tags=None, genre=None, count=10, delay=5):
    print(f"\n{'='*60}")
    print(f"Jamendo -> Mode B Batch Analyzer")
    print(f"Tags: {tags or 'any'} | Genre: {genre or 'any'} | Count: {count}")
    print(f"{'='*60}\n")

    tracks = search_tracks(tags=tags, genre=genre, limit=count)
    if not tracks:
        print("[ERROR] No tracks found.")
        return

    print(f"Found {len(tracks)} tracks\n")

    results = []
    for i, track in enumerate(tracks):
        print(f"[{i+1}/{len(tracks)}] {track['artist']} - {track['name']}")
        print(f"  Duration: {track['duration']}s | Genre: {', '.join(track['genre'][:3])}")

        audio_url = track["audio_url"] or track["audiodownload_url"]
        if not audio_url:
            print("  [SKIP] No audio URL")
            continue

        tmpfile = f"temp_jamendo_{track['id']}.mp3"
        dl = download_audio(audio_url, tmpfile)
        if not dl:
            continue

        label = f"jamendo_{track['id']}_{track['name'][:30]}"
        print(f"  Analyzing...")
        result = analyze_audio(tmpfile, label=label)

        try:
            os.remove(tmpfile)
        except:
            pass

        if result:
            result["jamendo_meta"] = {
                "track_id": track["id"],
                "name": track["name"],
                "artist": track["artist"],
                "genre": track["genre"],
                "tags": track["tags"],
                "license": track["license"],
            }
            results.append(result)

            predicted = result.get("predicted_prompt", {})
            prompt = predicted.get("predicted_prompt", "N/A") if isinstance(predicted, dict) else "N/A"
            score = result.get("quality_score", "N/A")
            reeval = result.get("reeval_recommended", False)

            print(f"  > Quality: {score} | Reeval: {reeval}")
            print(f"  > Predicted: {prompt[:80]}...")
        else:
            print(f"  X Analysis failed")

        if i < len(tracks) - 1:
            print(f"  Waiting {delay}s...")
            time.sleep(delay)

    # 결과 저장
    output_file = f"jamendo_analysis_{tags or genre or 'mixed'}_{len(results)}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done! {len(results)}/{len(tracks)} tracks analyzed")
    print(f"Results saved: {output_file}")

    scores = [r["quality_score"] for r in results if r.get("quality_score")]
    if scores:
        print(f"Quality scores: min={min(scores)}, max={max(scores)}, avg={sum(scores)/len(scores):.1f}")
        reeval_count = sum(1 for r in results if r.get("reeval_recommended"))
        print(f"Reeval recommended: {reeval_count}/{len(results)}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    # 사용법:
    # python jamendo_analyzer.py electronic 10
    # python jamendo_analyzer.py jazz 5
    # python jamendo_analyzer.py rock+indie 20
    # set JAMENDO_CLIENT_ID=your_key 먼저 실행

    tag = sys.argv[1] if len(sys.argv) > 1 else "electronic"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    run_batch(tags=tag, count=count, delay=5)
