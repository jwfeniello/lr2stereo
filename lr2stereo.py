import os
import sys
import re
import shutil
import numpy as np
import soundfile as sf
import concurrent.futures

# (l_suffix_regex, r_suffix_regex)
_PATTERNS = [
    (re.compile(r'^(.+) -L\.wav$', re.IGNORECASE), re.compile(r'^(.+) -R\.wav$', re.IGNORECASE)),
    (re.compile(r'^(.+)-L\.wav$', re.IGNORECASE),  re.compile(r'^(.+)-R\.wav$', re.IGNORECASE)),
    (re.compile(r'^(.+?)\s+L\.wav$', re.IGNORECASE), re.compile(r'^(.+?)\s+R\.wav$', re.IGNORECASE)),
]

def _extract_side(filename):
    for l_pat, r_pat in _PATTERNS:
        m = l_pat.match(filename)
        if m:
            return m.group(1).rstrip(), 'L'
        m = r_pat.match(filename)
        if m:
            return m.group(1).rstrip(), 'R'
    return None, None

def find_pairs(folder):
    pairs = []
    orphans = []

    for dirpath, _, filenames in os.walk(folder):
        candidates = {} 
        for fname in filenames:
            if not fname.lower().endswith('.wav'):
                continue
            base, side = _extract_side(fname)
            if base is None:
                continue
            if base not in candidates:
                candidates[base] = {}
            candidates[base][side] = os.path.join(dirpath, fname)

        for base, sides in candidates.items():
            if 'L' in sides and 'R' in sides:
                output = os.path.join(dirpath, base + '.wav')
                pairs.append({'l': sides['L'], 'r': sides['R'], 'base': base, 'output': output})
            else:
                for path in sides.values():
                    orphans.append(path)

    return pairs, orphans

def validate_pair(l_path, r_path):
    l_info = sf.info(l_path)
    r_info = sf.info(r_path)

    if l_info.samplerate != r_info.samplerate:
        return False, f"sample rate mismatch ({l_info.samplerate} vs {r_info.samplerate})"
    if l_info.channels != 1:
        return False, f"L file is not mono (has {l_info.channels} channels)"
    if r_info.channels != 1:
        return False, f"R file is not mono (has {r_info.channels} channels)"
    if l_info.subtype != r_info.subtype:
        return False, f"bit depth mismatch ({l_info.subtype} vs {r_info.subtype})"
    if l_info.frames != r_info.frames:
        return False, f"length mismatch ({l_info.frames} vs {r_info.frames} frames)"

    return True, ""

def merge_pair(l_path, r_path, output_path):
    l_info = sf.info(l_path)
    l_data, samplerate = sf.read(l_path, always_2d=False)
    r_data, _ = sf.read(r_path, always_2d=False)
    stereo = np.column_stack([l_data, r_data])
    sf.write(output_path, stereo, samplerate, subtype=l_info.subtype, format=l_info.format)

def process_single_task(pair, action_choice):
    """Worker function to process a single pair in its own thread."""
    is_valid, reason = validate_pair(pair['l'], pair['r'])

    if not is_valid:
        return "skipped", {'l': pair['l'], 'r': pair['r'], 'reason': reason}

    try:
        merge_pair(pair['l'], pair['r'], pair['output'])
        
        # Cleanup happens instantly inside the thread upon success
        if action_choice == "delete":
            for path in (pair["l"], pair["r"]):
                try:
                    os.remove(path)
                except OSError:
                    pass
        elif action_choice == "move":
            for path in (pair["l"], pair["r"]):
                old_dir = os.path.join(os.path.dirname(path), "_old")
                os.makedirs(old_dir, exist_ok=True)
                try:
                    shutil.move(path, os.path.join(old_dir, os.path.basename(path)))
                except OSError:
                    pass
                    
        return "merged", pair
    except Exception as e:
        return "error", {'output': pair['output'], 'error': str(e)}

def write_report(root, merged, skipped, orphans, errors):
    report_path = os.path.join(root, "stereo_merge_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Stereo Merge Report\n{'=' * 60}\n\n")

        f.write(f"MERGED ({len(merged)})\n{'-' * 40}\n")
        for m in merged:
            f.write(f"  {m['l']}\n  {m['r']}\n  -> {m['output']}\n\n")

        f.write(f"\nSKIPPED ({len(skipped)})\n{'-' * 40}\n")
        for s in skipped:
            f.write(f"  {s['l']}\n  {s['r']}\n  Reason: {s['reason']}\n\n")

        f.write(f"\nORPHANED FILES ({len(orphans)})\n{'-' * 40}\n")
        for o in orphans:
            f.write(f"  {o}\n")

        f.write(f"\nERRORS ({len(errors)})\n{'-' * 40}\n")
        for e in errors:
            f.write(f"  {e['output']}: {e['error']}\n")

    return report_path

def main():
    folder = os.getcwd()
    
    # Keeping this set to delete so you don't max out your storage
    ACTION_CHOICE = "delete" 

    print(f"Scanning directory: {folder}")

    pairs, orphans = find_pairs(folder)
    
    if not pairs:
        print("No matching L/R pairs found in the directory.")
        return

    merged, skipped, errors = [], [], []

    print(f"Found {len(pairs)} pairs. Hammering the disk with multiple threads now...\n")

    # Blast the tasks across available CPU threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks
        future_to_pair = {executor.submit(process_single_task, pair, ACTION_CHOICE): pair for pair in pairs}
        
        # Process results as they finish
        for count, future in enumerate(concurrent.futures.as_completed(future_to_pair), 1):
            status, result = future.result()
            
            if status == "merged":
                merged.append(result)
                print(f"[{count}/{len(pairs)}] Merged & Cleaned: {result['base']}")
            elif status == "skipped":
                skipped.append(result)
                print(f"[{count}/{len(pairs)}] Skipped: {result['reason']}")
            elif status == "error":
                errors.append(result)
                print(f"[{count}/{len(pairs)}] Error: {result['error']}")

    report_path = write_report(folder, merged, skipped, orphans, errors)
    print(f"\nDone. Report saved to: {report_path}")

if __name__ == "__main__":
    main()