import os
import numpy as np
import xml.etree.ElementTree as ET
import mne
import argparse

EPOCH_SEC = 30.0
MIN_CROP_START_SEC = 120
MIN_CROP_END_SEC = 120
EEG_CH_NAMES = ["EEG1", "EEG2", "EEG3"]
MAX_EPOCHS_PER_SUBJECT = 200
MAX_SUBJECTS = 400
POST_APNEA_WINDOW_SEC = 2.0
Z_THRESH = 6.0

EXCLUDE_KEYWORDS = [
    "Unsure", "unsure", "artifact", "Artifact",
    "Limb Movement", "limb", "Periodic leg movement", "PLM",
    "unreliable", "Unreliable"
]


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    for ev in root.findall(".//ScoredEvent"):
        ev_type = ev.findtext("EventConcept", "").strip()
        start = float(ev.findtext("Start"))
        dur = float(ev.findtext("Duration"))
        events.append({"type": ev_type, "start": start, "duration": dur})
    return events


def categorize_stage(ev_type):
    t = ev_type.lower()
    if "wake" in t: return 0
    if "stage 1" in t: return 1
    if "stage 2" in t: return 2
    if "stage 3" in t or "stage 4" in t: return 3
    if "rem" in t: return 5
    return None


def is_apnea_event(ev_type):
    t = ev_type.lower()
    return ("apnea" in t) or ("hypopnea" in t)


def is_exclude_event(ev_type):
    return any(kw in ev_type for kw in EXCLUDE_KEYWORDS)


def build_sample_labels(events, n_samples, sfreq):
    stage = np.full(n_samples, -1, dtype=np.int16)
    apnea = np.zeros(n_samples, dtype=np.uint8)
    exclude = np.zeros(n_samples, dtype=np.uint8)

    for ev in events:
        s = int(ev["start"] * sfreq)
        e = int((ev["start"] + ev["duration"]) * sfreq)
        s = max(0, min(s, n_samples))
        e = max(0, min(e, n_samples))
        if e <= s:
            continue

        ev_type = ev["type"]
        stage_code = categorize_stage(ev_type)

        if stage_code is not None:
            stage[s:e] = stage_code
        elif is_apnea_event(ev_type):
            apnea[s:e] = 1
        elif is_exclude_event(ev_type):
            exclude[s:e] = 1

    stage[stage == -1] = 0
    exclude[stage == 0] = 1

    return stage, apnea, exclude


def compute_crop(n_samples, sfreq):
    epoch_len = int(EPOCH_SEC * sfreq)
    start_idx = int(np.ceil(MIN_CROP_START_SEC / EPOCH_SEC)) * epoch_len
    end_idx = n_samples - int(np.ceil(MIN_CROP_END_SEC / EPOCH_SEC)) * epoch_len

    usable = end_idx - start_idx
    n_epochs = usable // epoch_len
    final_end = start_idx + n_epochs * epoch_len

    return start_idx, final_end, n_epochs, epoch_len


def epoch_data(raw, stage, apnea, exclude, start_idx, end_idx, n_epochs, epoch_len):
    data = raw.get_data(start=start_idx, stop=end_idx, picks=EEG_CH_NAMES)
    n_ch, _ = data.shape

    data_ep = data.reshape(n_ch, n_epochs, epoch_len).transpose(1, 0, 2)
    stage_ep = stage[start_idx:end_idx].reshape(n_epochs, epoch_len)
    apnea_ep = apnea[start_idx:end_idx].reshape(n_epochs, epoch_len)
    exclude_ep = exclude[start_idx:end_idx].reshape(n_epochs, epoch_len)

    stage_label = stage_ep[:, 0]
    exclude_label = (exclude_ep.max(axis=1) > 0).astype(np.uint8)
    apnea_frac = apnea_ep.mean(axis=1)

    return data_ep, stage_label, apnea_ep, apnea_frac, exclude_label


def qc_signal(data_norm, z_thresh=Z_THRESH, flat_std=1e-6):
    chan_std = data_norm.std(axis=2)
    flat = chan_std < flat_std
    bad_flat = flat.any(axis=1)

    max_abs = np.max(np.abs(data_norm), axis=(1, 2))
    bad_amp = max_abs > z_thresh

    good = ~(bad_flat | bad_amp)
    return good


def compute_apnea_flags(events, start_idx, end_idx, n_epochs, epoch_len, sfreq):
    apnea_events = [ev for ev in events if is_apnea_event(ev["type"])]

    apnea_end_flag = np.zeros(n_epochs, dtype=bool)
    post_apnea_flag = np.zeros(n_epochs, dtype=bool)

    if not apnea_events:
        return apnea_end_flag, post_apnea_flag

    post_window_samples = int(POST_APNEA_WINDOW_SEC * sfreq)
    eps = 1e-6

    for ev in apnea_events:
        end_sec = ev["start"] + ev["duration"]
        end_sample = end_sec * sfreq

        if end_sample <= start_idx or end_sample >= end_idx:
            continue

        rel_end = end_sample - start_idx
        end_epoch = int((rel_end - eps) // epoch_len)

        if 0 <= end_epoch < n_epochs:
            apnea_end_flag[end_epoch] = True

        post_start = max(end_sample, start_idx)
        post_end = min(end_sample + post_window_samples, end_idx - eps)

        if post_end <= post_start:
            continue

        rel_post_start = post_start - start_idx
        rel_post_end = post_end - start_idx

        first_ep = int(rel_post_start // epoch_len)
        last_ep = int((rel_post_end - eps) // epoch_len)

        for ep in range(first_ep, last_ep + 1):
            if 0 <= ep < n_epochs:
                post_apnea_flag[ep] = True

    return apnea_end_flag, post_apnea_flag


def compute_apnea_label(apnea_frac, apnea_end_flag, post_apnea_flag):
    n_epochs = len(apnea_frac)
    labels = np.full(n_epochs, -1, dtype=np.int8)

    no_apnea_mask = (apnea_frac == 0.0)
    labels[no_apnea_mask] = 0

    pos_mask = apnea_end_flag | (apnea_frac >= (10.0 / 30.0))
    labels[pos_mask] = 1

    post_only_mask = post_apnea_flag & (~apnea_end_flag) & no_apnea_mask
    labels[post_only_mask] = -1

    return labels


def stratified_sample(data, stage, labels, max_epochs, random_seed=42):
    n_total = data.shape[0]

    if n_total <= max_epochs:
        return data, stage, labels

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    if n_pos == 0 or n_neg == 0:
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(n_total, max_epochs, replace=False)
        indices = np.sort(indices)
        return data[indices], stage[indices], labels[indices]

    target_pos = int(max_epochs * (n_pos / (n_pos + n_neg)))
    target_neg = max_epochs - target_pos

    target_pos = min(target_pos, n_pos)
    target_neg = min(target_neg, n_neg)

    rng = np.random.RandomState(random_seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    sampled_pos = rng.choice(pos_idx, target_pos, replace=False)
    sampled_neg = rng.choice(neg_idx, target_neg, replace=False)

    indices = np.concatenate([sampled_pos, sampled_neg])
    indices = np.sort(indices)

    return data[indices], stage[indices], labels[indices]


def process_subject(edf_path, xml_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times

    events = parse_xml(xml_path)
    stage, apnea, exclude = build_sample_labels(events, n_samples, sfreq)

    start_idx, end_idx, n_epochs, epoch_len = compute_crop(n_samples, sfreq)

    data_ep, stage_ep, apnea_ep, apnea_frac, exclude_label = epoch_data(
        raw, stage, apnea, exclude, start_idx, end_idx, n_epochs, epoch_len
    )

    apnea_end_flag, post_apnea_flag = compute_apnea_flags(
        events, start_idx, end_idx, n_epochs, epoch_len, sfreq
    )

    keep = (exclude_label == 0)
    data_ep = data_ep[keep]
    stage_ep = stage_ep[keep]
    apnea_frac = apnea_frac[keep]
    apnea_end_flag = apnea_end_flag[keep]
    post_apnea_flag = post_apnea_flag[keep]

    mean = data_ep.mean(axis=(0, 2), keepdims=True)
    std = data_ep.std(axis=(0, 2), keepdims=True) + 1e-6
    data_norm = (data_ep - mean) / std

    qc_mask = qc_signal(data_norm)
    data_qc = data_norm[qc_mask]
    stage_qc = stage_ep[qc_mask]
    apnea_frac_qc = apnea_frac[qc_mask]
    apnea_end_qc = apnea_end_flag[qc_mask]
    post_apnea_qc = post_apnea_flag[qc_mask]

    apnea_label = compute_apnea_label(apnea_frac_qc, apnea_end_qc, post_apnea_qc)

    valid_mask = (apnea_label != -1)
    data_valid = data_qc[valid_mask]
    stage_valid = stage_qc[valid_mask]
    apnea_valid = apnea_label[valid_mask]

    data_final, stage_final, apnea_final = stratified_sample(
        data_valid, stage_valid, apnea_valid, MAX_EPOCHS_PER_SUBJECT
    )

    return data_final, stage_final, apnea_final


def get_all_subject_ids(edf_dir, xml_dir):
    edf_ids = set(f.replace(".edf", "").replace("mesa-sleep-", "")
                  for f in os.listdir(edf_dir) if f.endswith(".edf"))
    xml_ids = set(f.replace("-nsrr.xml", "").replace("mesa-sleep-", "")
                  for f in os.listdir(xml_dir) if f.endswith(".xml"))
    return sorted(edf_ids & xml_ids)


def run_all_subjects(base_dir, out_dir):
    edf_dir = os.path.join(base_dir, "polysomnography", "edfs")
    xml_dir = os.path.join(base_dir, "polysomnography", "annotations-events-nsrr")

    subject_ids = get_all_subject_ids(edf_dir, xml_dir)
    print(f"found {len(subject_ids)} subjects")

    if len(subject_ids) > MAX_SUBJECTS:
        subject_ids = subject_ids[:MAX_SUBJECTS]

    os.makedirs(out_dir, exist_ok=True)

    failed = []
    success = 0

    for i, sid in enumerate(subject_ids, 1):
        print(f"\n[{i}/{len(subject_ids)}] processing {sid}")

        edf_path = os.path.join(edf_dir, f"mesa-sleep-{sid}.edf")
        xml_path = os.path.join(xml_dir, f"mesa-sleep-{sid}-nsrr.xml")

        try:
            data, stage, apnea = process_subject(edf_path, xml_path)

            if data.shape[0] == 0:
                raise ValueError("no valid epochs")

            out_file = os.path.join(out_dir, f"mesa_{sid}.npz")
            np.savez(out_file, EEG=data, stage=stage, apnea=apnea)

            print(f"  saved {data.shape[0]} epochs")
            success += 1

        except Exception as e:
            print(f"  failed: {e}")
            failed.append((sid, str(e)))

    print(f"\nprocessed {success}/{len(subject_ids)} subjects")

    if failed:
        fail_log = os.path.join(out_dir, "failed_subjects.txt")
        with open(fail_log, "w") as f:
            for sid, error in failed:
                f.write(f"{sid}: {error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        run_all_subjects(args.base_dir, args.out_dir)
    elif args.subject:
        sid = args.subject.zfill(4)
        edf_path = os.path.join(args.base_dir, "polysomnography", "edfs", f"mesa-sleep-{sid}.edf")
        xml_path = os.path.join(args.base_dir, "polysomnography", "annotations-events-nsrr",
                                f"mesa-sleep-{sid}-nsrr.xml")

        data, stage, apnea = process_subject(edf_path, xml_path)

        os.makedirs(args.out_dir, exist_ok=True)
        out_file = os.path.join(args.out_dir, f"mesa_{sid}.npz")
        np.savez(out_file, EEG=data, stage=stage, apnea=apnea)

        print(f"saved {out_file}")
    else:
        parser.print_help()