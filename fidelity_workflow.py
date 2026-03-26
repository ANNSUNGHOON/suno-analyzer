"""
Fidelity Workflow - Semi-automatic pipeline
1. Read Jamendo analysis results - extract predicted prompts
2. Output prompt list for manual Suno generation
3. After Suno generation: analyze reproduction + compare with original
4. Generate fidelity report

Usage:
  Step 1: python fidelity_workflow.py prompts electronic
  Step 2: (manually generate in Suno, save mp3s to suno_outputs/ folder)
  Step 3: python fidelity_workflow.py compare electronic
"""

import os
import sys
import json
import glob
import requests
import time

ANALYZER_URL = os.getenv("ANALYZER_URL", "https://web-production-53cf6.up.railway.app")
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SUNO_OUTPUT_DIR = os.path.join(WORK_DIR, "suno_outputs")
RESULTS_DIR = os.path.join(WORK_DIR, "fidelity_results")


def load_analysis_results(genre: str) -> list:
    """Load Jamendo analysis JSON for a genre."""
    pattern = os.path.join(WORK_DIR, f"jamendo_analysis_{genre}_*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"No analysis files found for genre: {genre}")
        return []
    
    all_tracks = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            if isinstance(data, list):
                all_tracks.extend(data)
            elif isinstance(data, dict) and "results" in data:
                all_tracks.extend(data["results"])
    return all_tracks


def cmd_prompts(genre: str):
    """Step 1: Extract and display predicted prompts for Suno generation."""
    tracks = load_analysis_results(genre)
    if not tracks:
        return
    
    os.makedirs(SUNO_OUTPUT_DIR, exist_ok=True)
    
    prompt_list = []
    print(f"\n{'='*60}")
    print(f"  PREDICTED PROMPTS FOR SUNO - {genre.upper()}")
    print(f"{'='*60}\n")
    
    for i, track in enumerate(tracks, 1):
        predicted = track.get("predicted_prompt", {})
        prompt_text = predicted.get("predicted_prompt", "N/A") if isinstance(predicted, dict) else str(predicted)
        label_raw = track.get("label", "Unknown")
        parts = label_raw.split("_", 2)
        track_name = parts[2] if len(parts) > 2 else label_raw
        artist = ""
        analysis_id = track.get("analysis_id")
        quality = track.get("quality_score", 0)
        
        print(f"[{i}] {artist} - {track_name}")
        print(f"    Quality: {quality} | ID: {analysis_id}")
        print(f"    Prompt: {prompt_text}")
        print(f"    Save as: suno_outputs/{genre}_{i:02d}.mp3")
        print()
        
        prompt_list.append({
            "index": i,
            "track_name": track_name,
            "artist": artist,
            "analysis_id": analysis_id,
            "quality_score": quality,
            "predicted_prompt": prompt_text,
            "expected_filename": f"{genre}_{i:02d}.mp3"
        })
    
    prompt_file = os.path.join(WORK_DIR, f"prompts_{genre}.json")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompt_list, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*60}")
    print(f"  {len(prompt_list)} prompts saved to: prompts_{genre}.json")
    print(f"  Next: Generate each prompt in Suno")
    print(f"  Save mp3 files to: suno_outputs/{genre}_01.mp3, {genre}_02.mp3, ...")
    print(f"  Then run: python fidelity_workflow.py compare {genre}")
    print(f"{'='*60}")


def analyze_suno_audio(mp3_path: str, label: str) -> dict:
    """Upload Suno mp3 to /analyze-audio (Mode B) and return analysis result."""
    print(f"    Analyzing Suno output: {os.path.basename(mp3_path)} ...")
    with open(mp3_path, 'rb') as audio:
        files = {'file': (os.path.basename(mp3_path), audio, 'audio/mpeg')}
        data = {'label': label}
        r = requests.post(
            f"{ANALYZER_URL}/analyze-audio",
            files=files,
            data=data,
            timeout=300
        )
    if r.status_code == 200:
        result = r.json()
        aid = result.get("analysis_id")
        print(f"    -> analysis_id: {aid}")
        return result
    else:
        print(f"    -> ERROR {r.status_code}: {r.text[:200]}")
        return None


def compare_by_ids(original_id: int, reproduction_id: int, label: str = "") -> dict:
    """Call lightweight /compare endpoint with two analysis IDs."""
    print(f"    Comparing ID {original_id} vs {reproduction_id} ...")
    r = requests.post(
        f"{ANALYZER_URL}/compare",
        data={
            'original_id': str(original_id),
            'reproduction_id': str(reproduction_id),
            'label': label
        },
        timeout=60
    )
    if r.status_code == 200:
        return r.json()
    else:
        print(f"    -> ERROR {r.status_code}: {r.text[:200]}")
        return None


def cmd_compare(genre: str):
    """Step 3: Analyze Suno reproductions then compare with originals.
    
    New workflow (lightweight /compare):
    1. For each Suno mp3 -> POST /analyze-audio -> get reproduction_id
    2. POST /compare with original_id + reproduction_id
    3. Collect fidelity scores
    """
    prompt_file = os.path.join(WORK_DIR, f"prompts_{genre}.json")
    if not os.path.exists(prompt_file):
        print(f"No prompt file found. Run 'prompts {genre}' first.")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    
    print(f"\n{'='*60}")
    print(f"  FIDELITY COMPARISON - {genre.upper()}")
    print(f"{'='*60}\n")

    for p in prompts:
        mp3_path = os.path.join(SUNO_OUTPUT_DIR, p["expected_filename"])
        if not os.path.exists(mp3_path):
            print(f"  SKIP [{p['index']}] {p['expected_filename']} - file not found")
            continue
        
        original_id = p.get("analysis_id")
        if not original_id:
            print(f"  SKIP [{p['index']}] - no original analysis_id")
            continue
        
        print(f"  [{p['index']}] {p['track_name']} vs {p['expected_filename']}")
        
        try:
            # Step 1: Analyze Suno output with /analyze-audio
            label = f"suno_{genre}_{p['index']:02d}"
            suno_result = analyze_suno_audio(mp3_path, label)
            if not suno_result:
                continue
            
            reproduction_id = suno_result.get("analysis_id")
            if not reproduction_id:
                print(f"    -> No analysis_id returned, skipping")
                continue

            # Step 2: Compare original vs reproduction
            comp_label = f"{genre}_{p['index']:02d}_fidelity"
            comp = compare_by_ids(original_id, reproduction_id, comp_label)
            if not comp:
                continue
            
            fidelity = comp.get("fidelity", {})
            score = fidelity.get("fidelity_score", 0)
            verdict = fidelity.get("verdict", "unknown")
            weakest = fidelity.get("weakest", "?")
            strongest = fidelity.get("strongest", "?")
            breakdown = fidelity.get("breakdown", {})
            
            print(f"    Score: {score}/10 ({verdict})")
            print(f"    Best: {strongest} | Worst: {weakest}")
            dims = " | ".join(f"{k}:{v['grade']}" for k, v in breakdown.items())
            print(f"    {dims}")
            print()

            results.append({
                "index": p["index"],
                "track_name": p["track_name"],
                "original_id": original_id,
                "reproduction_id": reproduction_id,
                "predicted_prompt": p["predicted_prompt"],
                "fidelity_score": score,
                "verdict": verdict,
                "breakdown": breakdown,
                "weakest": weakest,
                "strongest": strongest,
                "comparison_id": comp.get("comparison_id")
            })
        
        except Exception as e:
            print(f"    ERROR: {e}")
        
        time.sleep(2)  # Rate limit between tracks

    # Save results and print report
    if results:
        out_file = os.path.join(RESULTS_DIR, f"fidelity_{genre}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump({
                "genre": genre,
                "total_compared": len(results),
                "avg_fidelity": round(sum(r["fidelity_score"] for r in results) / len(results), 1),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"  FIDELITY REPORT - {genre.upper()}")
        print(f"  Compared: {len(results)} tracks")
        avg = sum(r["fidelity_score"] for r in results) / len(results)
        print(f"  Average fidelity: {avg:.1f}/10")
        best = max(results, key=lambda x: x["fidelity_score"])
        worst = min(results, key=lambda x: x["fidelity_score"])
        print(f"  Best:  [{best['index']}] {best['track_name']} - {best['fidelity_score']}")
        print(f"  Worst: [{worst['index']}] {worst['track_name']} - {worst['fidelity_score']}")
        print(f"  Saved: {out_file}")
        print(f"{'='*60}")
    else:
        print("No comparisons completed.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python fidelity_workflow.py prompts <genre>")
        print("  python fidelity_workflow.py compare <genre>")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    genre = sys.argv[2].lower()
    
    if command == "prompts":
        cmd_prompts(genre)
    elif command == "compare":
        cmd_compare(genre)
    else:
        print(f"Unknown command: {command}")
        print("Use 'prompts' or 'compare'")
