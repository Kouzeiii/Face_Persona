#!/usr/bin/env python3
"""
FacePersona Research Maintenance CLI
Provides direct command-line administration for researchers on the local server.
Permits system status inspection, model retraining, subject deletion, and image
ledger auditing without exposing remote endpoints or requiring network requests.
"""

import os
import sys
import json
import argparse
import pandas as pd

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import app


def print_banner():
    print("=" * 65)
    print("  FacePersona Research Maintenance CLI")
    print("  Direct Server Administrative Interface")
    print("=" * 65)


def cmd_status(args):
    """Display comprehensive system, dataset, model, and storage status."""
    print("\n--- System & Dataset Status ---")
    
    # 1. Dataset records
    emb_count = 0
    if os.path.exists(app.FACE_EMBEDDINGS_FILE):
        try:
            df = pd.read_csv(app.FACE_EMBEDDINGS_FILE)
            emb_count = len(df)
        except Exception:
            pass

    score_count = 0
    if os.path.exists(app.PERSONALITY_SCORES_FILE):
        try:
            df = pd.read_csv(app.PERSONALITY_SCORES_FILE)
            score_count = len(df)
        except Exception:
            pass

    dataset_count = 0
    if os.path.exists(app.DATASET_FILE):
        try:
            df = pd.read_csv(app.DATASET_FILE)
            dataset_count = len(df)
        except Exception:
            pass

    profile_count = 0
    if os.path.exists(app.PROFILES_DIR):
        profile_count = len([f for f in os.listdir(app.PROFILES_DIR) if f.endswith('.json')])

    upload_count = 0
    if os.path.exists(app.UPLOAD_DIR):
        upload_count = len([f for f in os.listdir(app.UPLOAD_DIR) if os.path.isfile(os.path.join(app.UPLOAD_DIR, f))])

    print(f"  Paired Dataset Records (dataset.csv)    : {dataset_count}")
    print(f"  Face Embeddings (face_embeddings.csv)    : {emb_count}")
    print(f"  Personality Scores (personality_scores) : {score_count}")
    print(f"  Profile JSON Files (data/profiles/)     : {profile_count}")
    print(f"  Stored Upload Images (uploads/)         : {upload_count}")

    # 2. Model status
    print("\n--- Machine Learning Model Status ---")
    model_exists = os.path.exists(app.TRAINED_MODEL_FILE)
    print(f"  Trained Model File (trained_model.pkl)  : {'Present' if model_exists else 'NOT FOUND'}")

    is_outdated, outdated_reason = app.get_model_outdated_status()
    print(f"  Model Up-to-Date Status                 : {'OUTDATED' if is_outdated else 'Current'}")
    if is_outdated:
        print(f"  Outdated Reason                         : {outdated_reason}")

    if os.path.exists(app.MODEL_METRICS_FILE):
        try:
            with open(app.MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            print(f"  Last Training Date                      : {metrics.get('training_date', 'Unknown')}")
            print(f"  Evaluation Subjects (Train/Test)        : {metrics.get('n_train', '?')} / {metrics.get('n_test', '?')} (Total: {metrics.get('n_subjects', '?')})")
            print(f"  PCA Components                          : {metrics.get('pca_components', '?')}")
            print(f"  OOB Score (R2)                          : {metrics.get('oob_score', 'N/A')}")
            
            print("\n  Held-Out Test Set Performance:")
            tm = metrics.get('trait_metrics', {})
            print(f"    {'Trait':<18} {'MAE':<8} {'R2':<10} {'Pearson r':<12} {'p-value':<10}")
            print("    " + "-" * 58)
            for trait in app.TRAIT_NAMES:
                data = tm.get(trait, {})
                mae = data.get('mae', 'N/A')
                r2 = data.get('r2')
                r2_str = f"{r2:.4f}" if r2 is not None else "null"
                pr = data.get('pearson_r')
                pr_str = f"{pr:.4f}" if pr is not None else "null"
                pv = data.get('p_value')
                pv_str = f"{pv:.4f}" if pv is not None else "null"
                print(f"    {app.TRAIT_LABELS.get(trait, trait):<18} {mae:<8} {r2_str:<10} {pr_str:<12} {pv_str:<10}")
        except Exception as e:
            print(f"  Error reading model metrics: {e}")

    # 3. Environment status
    print("\n--- Environment Configuration ---")
    admin_key_set = bool(os.environ.get('RESEARCH_ADMIN_KEY'))
    claude_key_set = bool(os.environ.get('ANTHROPIC_API_KEY') and os.environ.get('ANTHROPIC_API_KEY') not in app.PLACEHOLDER_API_KEYS)
    print(f"  RESEARCH_ADMIN_KEY Configured           : {'Yes' if admin_key_set else 'No (Remote admin calls locked)'}")
    print(f"  ANTHROPIC_API_KEY Configured            : {'Yes (Active)' if claude_key_set else 'No / Placeholder'}")
    print(f"  Sonnet Model Identifier                 : {app.CLAUDE_SONNET_MODEL}")
    print(f"  Haiku Model Identifier                  : {app.CLAUDE_HAIKU_MODEL}")
    print("=" * 65 + "\n")


def cmd_retrain(args):
    """Execute model retraining directly."""
    print("\n[INFO] Commencing model retraining pipeline...")
    success, message, metrics, updated = app.perform_retrain_model()
    if not success:
        print(f"\n[ERROR] Retraining failed: {message}")
        sys.exit(1)

    print(f"\n[SUCCESS] {message}")
    if metrics:
        print("\nUpdated Held-Out Evaluation Metrics:")
        tm = metrics.get('trait_metrics', {})
        print(f"  {'Trait':<18} {'MAE':<8} {'R2':<10} {'Pearson r':<12} {'p-value':<10}")
        print("  " + "-" * 58)
        for trait in app.TRAIT_NAMES:
            data = tm.get(trait, {})
            mae = data.get('mae', 'N/A')
            r2 = data.get('r2')
            r2_str = f"{r2:.4f}" if r2 is not None else "null"
            pr = data.get('pearson_r')
            pr_str = f"{pr:.4f}" if pr is not None else "null"
            pv = data.get('p_value')
            pv_str = f"{pv:.4f}" if pv is not None else "null"
            print(f"  {app.TRAIT_LABELS.get(trait, trait):<18} {mae:<8} {r2_str:<10} {pr_str:<12} {pv_str:<10}")
    print()


def cmd_delete(args):
    """Safely delete a subject by name."""
    name = args.name.strip()
    if not name:
        print("[ERROR] Subject name must be provided.")
        sys.exit(1)

    print(f"\n[INFO] Attempting safe deletion of subject: '{name}'...")
    result, status_code = app.perform_delete_subject(name)

    if status_code == 200:
        print(f"[SUCCESS] {result.get('message')}")
        if result.get('model_outdated'):
            print("[NOTICE] Trained model is now flagged as outdated. Run 'python manage_research.py retrain' to update.")
    elif status_code == 207:
        print(f"[WARNING] Partial deletion completed with warnings (HTTP 207):")
        print(f"  {result.get('message')}")
        for w in result.get('warnings', []):
            print(f"  - {w}")
    elif status_code == 404:
        print(f"[ERROR] Subject not found (HTTP 404): {result.get('error')}")
        sys.exit(1)
    else:
        print(f"[ERROR] Deletion failed (HTTP {status_code}): {result.get('error')}")
        sys.exit(1)
    print()


def cmd_audit_images(args):
    """Audit stored image files, subject references, and unresolved legacy records."""
    print("\n--- Image Storage & Subject Ledger Audit ---")
    if os.path.exists(app.UNRESOLVED_IMAGES_REPORT_FILE):
        try:
            with open(app.UNRESOLVED_IMAGES_REPORT_FILE, 'r', encoding='utf-8') as f:
                report = json.load(f)
            summary = report.get('summary', {})
            total_subjects = 103
            if os.path.exists(app.FACE_EMBEDDINGS_FILE):
                try:
                    df = pd.read_csv(app.FACE_EMBEDDINGS_FILE)
                    total_subjects = len(df)
                except Exception:
                    pass

            verified_count = summary.get('verified_subject_images', 0)
            unresolved_count = summary.get('unresolved_subjects_count', 0)
            coverage = round((verified_count / total_subjects) * 100, 1) if total_subjects > 0 else 0.0

            print(f"  Audit Generated At                      : {report.get('audit_timestamp', 'Unknown')}")
            print(f"  Total Subjects in Dataset               : {total_subjects}")
            print(f"  Total Files in uploads/                 : {summary.get('total_upload_files', 0)}")
            print(f"  Verified Associated Subjects            : {verified_count}")
            print(f"  Unresolved Legacy Subjects              : {unresolved_count}")
            print(f"  Ambiguous Candidate Images              : {summary.get('ambiguous_candidate_images', 0)}")
            print(f"  Unreferenced Orphan Upload Images       : {summary.get('orphaned_images_count', 0)}")
            print(f"  Resolved Verified Coverage              : {coverage}%")

            unresolved = report.get('unresolved_subjects', [])
            if unresolved:
                print(f"\n  Unresolved Legacy Subjects ({len(unresolved)}):")
                for u in unresolved:
                    sub = u.get('name')
                    status = u.get('status')
                    cands = u.get('candidate_images', [])
                    print(f"    - {sub:<25} [Status: {status}] Candidates: {cands or 'None'}")

            orphaned = report.get('orphaned_image_files', [])
            if orphaned:
                print(f"\n  Unreferenced Upload Files ({len(orphaned)}):")
                for o in orphaned:
                    print(f"    - {o}")
        except Exception as e:
            print(f"  Error reading audit report: {e}")
    else:
        print("  Audit report file not found. Run analysis or generate audit report.")
    print("=" * 65 + "\n")


def cmd_private_list(args):
    """List private participants (no names are stored)."""
    path = app.get_private_participants_file()
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("\nNo private participants recorded yet.")
        return
    df = pd.read_csv(path)
    cols = ['participant_id', 'consent_timestamp', 'age_range', 'openness', 'conscientiousness',
            'extraversion', 'agreeableness', 'neuroticism']
    print(f"\n--- Private Participants ({len(df)}) ---")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


def cmd_private_delete(args):
    """Delete a private participant's record and photo (data deletion request)."""
    ok, msg = app.delete_private_participant(args.participant_id)
    print(("\n✓ " if ok else "\n") + msg)
    if not ok:
        sys.exit(1)


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        description="FacePersona Research Maintenance CLI - Direct server management tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available administrative commands")

    # Command: status
    subparsers.add_parser("status", help="Show system, dataset, model, and storage status")

    # Command: retrain
    subparsers.add_parser("retrain", help="Retrain machine learning model on current dataset")

    # Command: delete
    del_parser = subparsers.add_parser("delete", help="Safely delete a subject and associated data")
    del_parser.add_argument("name", help="Name of the subject to delete")

    # Command: audit-images
    subparsers.add_parser("audit-images", help="Display image association ledger and unresolved report")

    # Commands: private participants
    subparsers.add_parser("private-list", help="List private participants (Participant IDs, age range, scores)")
    pdel = subparsers.add_parser("private-delete", help="Delete a private participant's data (deletion request)")
    pdel.add_argument("participant_id", help="Participant ID, e.g. P-1A2B3C4D")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "retrain":
        cmd_retrain(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "audit-images":
        cmd_audit_images(args)
    elif args.command == "private-list":
        cmd_private_list(args)
    elif args.command == "private-delete":
        cmd_private_delete(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
