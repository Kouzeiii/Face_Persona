"""
FacePersona — Safe Deployment Packaging & Verification Utility
Author: FacePersona Deployment & Security Team
Date: October 2026

Creates sanitized, production-ready deployment archives with mathematically
enforced exclusion of sensitive credentials (.env, API keys, secrets, databases),
temporary files, Python caches, and testing suites.
"""

import os
import sys
import fnmatch
import zipfile
import shutil
import argparse
from pathlib import Path

# Strict exclusion patterns: credentials, caches, databases, archives, and tests
EXCLUDED_PATTERNS = [
    # Sensitive credentials and local environment files
    '.env',
    '.env.*',
    '.env.local',
    '.env.*.local',
    '*.pem',
    '*.key',
    '*.secret',
    'credentials.*',
    '*.lock',

    # Databases and local runtime data
    '*.db',
    '*.db-wal',
    '*.db-shm',
    '*.sqlite',
    '*.sqlite3',

    # Backups and corrupted files
    '*.bak',
    '*.bak.*',
    '*.corrupt',
    '*.corrupt.*',

    # Python caches and bytecode
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '*$py.class',
    '.Python',

    # Virtual environments
    'venv',
    'env',
    'ENV',
    '.venv',

    # Existing archive files
    '*.zip',
    '*.tar.gz',
    '*.tar',
    '*.7z',
    '*.rar',

    # IDE and OS metadata
    '.git',
    '.DS_Store',
    'Thumbs.db',
    '.idea',
    '.vscode',
    '*.swp',

    # Development-only testing suites
    'tests',
]

REQUIRED_DEPLOYMENT_FILES = [
    'app.py',
    'Procfile',
    'requirements.txt',
    '.env.example',
    'templates/base.html',
    'templates/index.html',
    'templates/analyze.html',
    'templates/profile.html',
    'templates/subjects.html',
    'templates/insights.html',
    'templates/restricted_profile.html',
    'static/css/style.css',
    'static/js/main.js',
]


def is_path_excluded(rel_path):
    """Determine whether a relative file path matches any exclusion rule.
    Explicitly whitelists non-sensitive .env.example template."""
    normalized = rel_path.replace('\\', '/')
    parts = normalized.split('/')

    # Always preserve .env.example as the safe public deployment template
    if parts[-1] in ('.env.example', 'env.example'):
        return False, None

    # 1. Exact or wildcard match on any individual directory component
    for part in parts:
        for pat in EXCLUDED_PATTERNS:
            if fnmatch.fnmatch(part, pat):
                return True, pat

    # 2. Match against full relative path
    for pat in EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(normalized, pat) or fnmatch.fnmatch(parts[-1], pat):
            return True, pat

    return False, None


def audit_project(source_dir):
    """Scan source directory and categorize files into included and safely excluded."""
    source_dir = os.path.abspath(source_dir)
    included = []
    excluded = []
    issues = []

    for root, dirs, files in os.walk(source_dir):
        # Prune excluded directories from walking
        rel_root = os.path.relpath(root, source_dir)
        if rel_root != '.':
            excl, pat = is_path_excluded(rel_root)
            if excl:
                excluded.append((rel_root + '/', pat))
                dirs[:] = []
                continue

        for fname in files:
            full_path = os.path.join(root, fname)
            rel_file = os.path.relpath(full_path, source_dir).replace('\\', '/')
            excl, pat = is_path_excluded(rel_file)
            if excl:
                excluded.append((rel_file, pat))
            else:
                included.append(rel_file)

    # Verify required deployment files exist in included list
    for req in REQUIRED_DEPLOYMENT_FILES:
        if req not in included:
            issues.append(f"Missing required deployment file: {req}")

    # Verify .env is strictly excluded
    for inc in included:
        if inc == '.env' or inc.endswith('/.env'):
            issues.append(f"SECURITY CRITICAL: .env file found in inclusion list: {inc}")

    return included, excluded, issues


def create_safe_zip(source_dir, output_zip_path):
    """Create a sanitized ZIP archive containing only authorized production deployment files."""
    source_dir = os.path.abspath(source_dir)
    output_zip_path = os.path.abspath(output_zip_path)

    included, excluded, issues = audit_project(source_dir)
    if issues:
        raise ValueError(f"Packaging aborted due to validation issues: {issues}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel_file in included:
            full_path = os.path.join(source_dir, rel_file)
            zf.write(full_path, arcname=rel_file)

    # Post-packaging security verification
    with zipfile.ZipFile(output_zip_path, 'r') as zf:
        namelist = zf.namelist()
        for name in namelist:
            excl, pat = is_path_excluded(name)
            if excl:
                os.remove(output_zip_path)
                raise RuntimeError(f"Security check failed! Excluded file {name} ({pat}) present in archive.")
            if name.endswith('.env') or '/.env' in name or name == '.env':
                os.remove(output_zip_path)
                raise RuntimeError(f"Security check failed! .env present in archive: {name}")

    return len(included), len(excluded)


def main():
    parser = argparse.ArgumentParser(description="FacePersona Safe Deployment Packaging Utility")
    parser.add_argument('--check', action='store_true', help="Audit project and report included/excluded files without creating an archive")
    parser.add_argument('--zip', type=str, default=None, help="Target ZIP archive path for safe packaging")
    parser.add_argument('--source', type=str, default='.', help="Source project directory (default: current directory)")

    args = parser.parse_args()
    source_dir = os.path.abspath(args.source)

    print("=================================================================")
    print("  FacePersona Safe Deployment Packaging & Verification")
    print("=================================================================")
    print(f"Source Directory: {source_dir}")

    included, excluded, issues = audit_project(source_dir)

    print(f"\nAudit Summary:")
    print(f"  Total Deployment Files Approved : {len(included)}")
    print(f"  Sensitive / Transient Excluded  : {len(excluded)}")
    print(f"  Configuration Issues Detected   : {len(issues)}")

    if issues:
        print("\n[!] CRITICAL ISSUES DETECTED:")
        for iss in issues:
            print(f"    - {iss}")
        sys.exit(1)

    print("\nExcluded Sensitive / Transient Items (Sampling):")
    for item, pat in excluded[:12]:
        print(f"  [X] {item} (Rule: {pat})")
    if len(excluded) > 12:
        print(f"  ... and {len(excluded) - 12} other items safely excluded.")

    if args.zip:
        print(f"\nBuilding safe deployment package: {args.zip}")
        n_inc, n_exc = create_safe_zip(source_dir, args.zip)
        print(f"[OK] Deployment package created successfully: {args.zip}")
        print(f"     Verified: 0 credentials, 0 .env files, 0 tests included.")
    else:
        print("\n[OK] Audit PASSED. Project is cleanly structured for safe packaging.")
        print("     To generate a verified ZIP package, run:")
        print("     python package_deployment.py --zip deploy.zip")


if __name__ == '__main__':
    main()
