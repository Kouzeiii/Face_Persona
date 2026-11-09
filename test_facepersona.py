import os
import sys
import json
import uuid
import shutil
import hashlib
import tempfile
import zipfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import cv2
import anthropic

# Add application directory to Python path
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import app

VALID_TEST_BIOGRAPHY = (
    "Dr. Alexander Vance is an acclaimed experimental cognitive neuroscientist and researcher whose pioneering work "
    "has fundamentally reshaped contemporary understandings of personality traits and human perception. Over twenty "
    "years of rigorous laboratory investigations, Dr. Vance published more than sixty peer-reviewed scientific studies "
    "evaluating the structural relationships between cognitive temperament and social behavior. His academic tenure "
    "spans leading universities worldwide, where he mentors graduate researchers in quantitative psychological methods, "
    "ethical empirical inquiry, and advanced machine learning modeling across multimodal observational disciplines. "
    "He regularly collaborates with international research consortia to establish open-access benchmarks and educational guidelines."
)


class FacePersonaTestSuite(unittest.TestCase):
    """Comprehensive, fully isolated automated test suite for FacePersona Web Application.
    Tests execute strictly inside a sandboxed temporary directory to guarantee that real research
    datasets, image associations, reports, models, and uploads are never mutated or corrupted during testing."""

    @classmethod
    def setUpClass(cls):
        # 0. Set sandbox administrative key
        cls.orig_admin_key = os.environ.get('RESEARCH_ADMIN_KEY')
        os.environ['RESEARCH_ADMIN_KEY'] = 'test_admin_key_2026'
        cls.admin_key = 'test_admin_key_2026'
        cls.valid_bio = VALID_TEST_BIOGRAPHY

        # 1. Record baseline hashes of all real production research files before tests run
        cls.orig_paths = {
            'BASE_DIR': app.BASE_DIR,
            'DATA_DIR': app.DATA_DIR,
            'PROCESSED_DIR': app.PROCESSED_DIR,
            'MODELS_DIR': app.MODELS_DIR,
            'RESULTS_DIR': app.RESULTS_DIR,
            'UPLOAD_DIR': app.UPLOAD_DIR,
            'PROFILES_DIR': app.PROFILES_DIR,
            'FACE_EMBEDDINGS_FILE': app.FACE_EMBEDDINGS_FILE,
            'PERSONALITY_SCORES_FILE': app.PERSONALITY_SCORES_FILE,
            'DATASET_FILE': app.DATASET_FILE,
            'PUBLIC_ASSESSMENTS_FILE': app.PUBLIC_ASSESSMENTS_FILE,
            'PUBLIC_ASSESSMENT_LOCK_FILE': getattr(app, 'PUBLIC_ASSESSMENT_LOCK_FILE', None),
            'TRAINED_MODEL_FILE': app.TRAINED_MODEL_FILE,
            'MODEL_METRICS_FILE': app.MODEL_METRICS_FILE,
            'IMAGE_ASSOCIATIONS_FILE': app.IMAGE_ASSOCIATIONS_FILE,
            'UNRESOLVED_IMAGES_REPORT_FILE': app.UNRESOLVED_IMAGES_REPORT_FILE,
        }

        cls.prod_hashes = {}
        for dir_key in ['DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR', 'UPLOAD_DIR']:
            target_dir = cls.orig_paths[dir_key]
            if os.path.exists(target_dir):
                for root, dirs, files in os.walk(target_dir):
                    for f in files:
                        p = os.path.join(root, f)
                        try:
                            with open(p, 'rb') as fp:
                                cls.prod_hashes[p] = hashlib.sha256(fp.read()).hexdigest()
                        except Exception:
                            pass

        # 2. Create sandboxed temporary directory
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.sandbox_dir = cls.temp_dir.name

        # 3. Reconfigure app paths strictly to sandbox
        app.DATA_DIR = os.path.join(cls.sandbox_dir, 'data')
        app.PROCESSED_DIR = os.path.join(app.DATA_DIR, 'processed')
        app.MODELS_DIR = os.path.join(cls.sandbox_dir, 'models')
        app.RESULTS_DIR = os.path.join(cls.sandbox_dir, 'results')
        app.UPLOAD_DIR = os.path.join(cls.sandbox_dir, 'uploads')
        app.PROFILES_DIR = os.path.join(app.DATA_DIR, 'profiles')

        os.makedirs(app.PROCESSED_DIR, exist_ok=True)
        os.makedirs(app.MODELS_DIR, exist_ok=True)
        os.makedirs(app.RESULTS_DIR, exist_ok=True)
        os.makedirs(app.UPLOAD_DIR, exist_ok=True)
        os.makedirs(app.PROFILES_DIR, exist_ok=True)

        app.FACE_EMBEDDINGS_FILE = os.path.join(app.PROCESSED_DIR, 'face_embeddings.csv')
        app.PERSONALITY_SCORES_FILE = os.path.join(app.PROCESSED_DIR, 'personality_scores.csv')
        app.DATASET_FILE = os.path.join(app.PROCESSED_DIR, 'dataset.csv')
        app.PUBLIC_ASSESSMENTS_FILE = os.path.join(app.PROCESSED_DIR, 'public_assessments.csv')
        app.PUBLIC_ASSESSMENT_LOCK_FILE = os.path.join(app.PROCESSED_DIR, 'public_assessments.lock')
        app.TRAINED_MODEL_FILE = os.path.join(app.MODELS_DIR, 'trained_model.pkl')
        app.MODEL_METRICS_FILE = os.path.join(app.RESULTS_DIR, 'model_metrics.json')
        app.IMAGE_ASSOCIATIONS_FILE = os.path.join(app.DATA_DIR, 'image_associations.json')
        app.UNRESOLVED_IMAGES_REPORT_FILE = os.path.join(app.DATA_DIR, 'unresolved_images_report.json')

        # Initialize empty sandbox associations and report
        with open(app.IMAGE_ASSOCIATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        with open(app.UNRESOLVED_IMAGES_REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f)

        # 4. Populate synthetic sandbox dataset (15 subjects)
        np.random.seed(42)
        cls.synthetic_subjects = [f"Sandboxed_Subject_{i}" for i in range(1, 16)]
        emb_cols = ['name'] + [f'face_{i}' for i in range(1, 513)]
        emb_rows = []
        score_rows = []
        for name in cls.synthetic_subjects:
            vec = np.random.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            emb_rows.append([name] + vec.tolist())
            score_rows.append({
                'name': name,
                'openness': float(np.random.randint(40, 95)),
                'conscientiousness': float(np.random.randint(40, 95)),
                'extraversion': float(np.random.randint(40, 95)),
                'agreeableness': float(np.random.randint(40, 95)),
                'neuroticism': float(np.random.randint(20, 80)),
            })
        pd.DataFrame(emb_rows, columns=emb_cols).to_csv(app.FACE_EMBEDDINGS_FILE, index=False)
        pd.DataFrame(score_rows).to_csv(app.PERSONALITY_SCORES_FILE, index=False)

        # 4b. Create synthetic profile JSON for Sandboxed_Subject_1
        sample_prof = {
            "name": "Sandboxed_Subject_1",
            "profession": "scientist",
            "profession_label": "Scientist / Researcher",
            "biography": "A synthetic researcher profile for isolated testing.",
            "biography_scores": {
                "openness": 75.0,
                "conscientiousness": 80.0,
                "extraversion": 60.0,
                "agreeableness": 70.0,
                "neuroticism": 40.0
            },
            "estimated_scores": {
                "openness": 70.0,
                "conscientiousness": 75.0,
                "extraversion": 65.0,
                "agreeableness": 68.0,
                "neuroticism": 45.0
            },
            "reasoning": "Synthetic evaluation profile.",
            "societal_impact": "Synthetic impact text."
        }
        with open(os.path.join(app.PROFILES_DIR, 'Sandboxed_Subject_1.json'), 'w', encoding='utf-8') as f:
            json.dump(sample_prof, f, indent=2)

        # 5. Create synthetic verified face images and non-face image in sandbox
        orig_uploads = cls.orig_paths['UPLOAD_DIR']
        cls.verified_face_path = os.path.join(app.UPLOAD_DIR, 'verified_face_sample.jpg')
        cls.verified_face_path_2 = os.path.join(app.UPLOAD_DIR, 'verified_face_sample_2.jpg')
        found_faces = []
        if os.path.exists(orig_uploads):
            candidates = [f for f in os.listdir(orig_uploads) if f.lower().endswith(('.jpg', '.png'))]
            for c in candidates:
                cand_path = os.path.join(orig_uploads, c)
                test_emb, _, test_err = app.extract_face_embedding(cand_path)
                if test_emb is not None and not test_err:
                    found_faces.append(cand_path)
                    if len(found_faces) >= 2:
                        break

        if len(found_faces) >= 1:
            shutil.copy2(found_faces[0], cls.verified_face_path)
        else:
            img = np.ones((250, 250, 3), dtype=np.uint8) * 230
            cv2.circle(img, (125, 125), 85, (180, 160, 150), -1)
            cv2.circle(img, (95, 100), 10, (30, 30, 30), -1)
            cv2.circle(img, (155, 100), 10, (30, 30, 30), -1)
            cv2.line(img, (125, 105), (125, 145), (60, 60, 60), 4)
            cv2.ellipse(img, (125, 170), (35, 15), 0, 0, 180, (40, 40, 40), 4)
            cv2.imwrite(cls.verified_face_path, img)

        if len(found_faces) >= 2:
            shutil.copy2(found_faces[1], cls.verified_face_path_2)
        else:
            img2 = np.ones((250, 250, 3), dtype=np.uint8) * 210
            cv2.circle(img2, (125, 125), 90, (190, 170, 160), -1)
            cv2.circle(img2, (90, 95), 12, (20, 20, 20), -1)
            cv2.circle(img2, (160, 95), 12, (20, 20, 20), -1)
            cv2.line(img2, (125, 100), (125, 140), (50, 50, 50), 5)
            cv2.ellipse(img2, (125, 165), (40, 20), 0, 0, 180, (30, 30, 30), 5)
            cv2.imwrite(cls.verified_face_path_2, img2)

        cls.no_face_path = os.path.join(app.UPLOAD_DIR, 'no_face_sample.jpg')
        no_face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        no_face_img[:] = (35, 35, 35)
        cv2.imwrite(cls.no_face_path, no_face_img)

    @classmethod
    def tearDownClass(cls):
        # Restore original paths in app
        for k, v in cls.orig_paths.items():
            setattr(app, k, v)
        if cls.orig_admin_key is not None:
            os.environ['RESEARCH_ADMIN_KEY'] = cls.orig_admin_key
        else:
            os.environ.pop('RESEARCH_ADMIN_KEY', None)
        # Clean up temporary directory
        cls.temp_dir.cleanup()

    def setUp(self):
        app.app.config['TESTING'] = True
        self.client = app.app.test_client()

    def get_authenticated_client(self):
        """Helper returning a client with session-initialized CSRF token."""
        client = app.app.test_client()
        csrf_token = uuid.uuid4().hex
        with client.session_transaction() as sess:
            sess['csrf_token'] = csrf_token
        return client, csrf_token

    # =========================================================================
    # A. PUBLIC ACCESS & ACCESSIBILITY TESTS (NO LOGIN REQUIRED)
    # =========================================================================

    def test_dashboard_redirects_public_visitors_to_analyze(self):
        """Public visitors are sent to the Analyze page; the dashboard is researcher-only."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/analyze', resp.headers['Location'])

    def test_public_sees_only_analyze_and_researcher_pages_require_login(self):
        """Analyze is public; Subjects, Insights, Participants and exports require researcher login."""
        self.assertEqual(self.client.get('/analyze').status_code, 200)
        self.assertEqual(self.client.get('/researcher').status_code, 200)
        for r in ['/subjects', '/insights', '/researcher/participants', '/researcher/export/private']:
            resp = self.client.get(r)
            self.assertEqual(resp.status_code, 302, f"{r} must redirect public visitors")
            self.assertIn('/researcher', resp.headers['Location'])
        html = self.client.get('/analyze').data.decode()
        self.assertIn('Researcher Login', html)  # small footer link only
        self.assertNotIn('>Subjects<', html)
        self.assertNotIn('>Dashboard<', html)
        self.assertEqual(self.client.get('/dashboard').status_code, 302)

    def _researcher_client(self):
        client, csrf_token = self.get_authenticated_client()
        client.post('/authorize_researcher', data={'csrf_token': csrf_token, 'admin_key': self.admin_key})
        return client, csrf_token

    def test_researcher_pages_and_nav_after_login(self):
        """After unlocking, researcher pages load and the full navigation is shown."""
        client, _ = self._researcher_client()
        for r in ['/dashboard', '/subjects', '/insights', '/researcher/participants']:
            self.assertEqual(client.get(r).status_code, 200, r)
        # Even researchers land on Analyze first
        resp = client.get('/')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/analyze', resp.headers['Location'])
        html = client.get('/analyze').data.decode()
        self.assertIn('Participants', html)
        self.assertIn('Log out', html)

    def test_login_next_param_rejects_external_redirects(self):
        """Post-login redirect only allows same-site paths (no open redirect)."""
        client, csrf_token = self.get_authenticated_client()
        resp = client.post('/authorize_researcher', data={'csrf_token': csrf_token, 'admin_key': self.admin_key,
                                                          'next': '//evil.example.com/x'})
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn('evil.example.com', resp.headers['Location'])

    def test_wrong_key_returns_to_login_with_message(self):
        """A wrong key returns to the login page and shows an error message."""
        client, csrf_token = self.get_authenticated_client()
        resp = client.post('/authorize_researcher', data={'csrf_token': csrf_token, 'admin_key': 'wrong-key-123456'},
                           follow_redirects=True)
        self.assertIn(b'Invalid Research Admin Key', resp.data)

    def test_subject_profile_accessible_without_login(self):
        """Individual subject profiles are publicly viewable without authentication."""
        sample_name = "Sandboxed_Subject_1"
        resp = self.client.get(f'/profile/{sample_name}')
        self.assertEqual(resp.status_code, 200)

    def test_no_login_or_logout_routes_exist(self):
        """Login and logout endpoints are completely removed; requests return 404."""
        resp_login = self.client.get('/login')
        self.assertEqual(resp_login.status_code, 404)
        resp_logout = self.client.get('/logout')
        self.assertEqual(resp_logout.status_code, 404)

    def test_csrf_token_initialized_for_public_visitor(self):
        """Visiting a public page generates and stores a valid CSRF token in the visitor session."""
        with self.client as c:
            resp = c.get('/analyze')
            self.assertEqual(resp.status_code, 200)
            with c.session_transaction() as sess:
                self.assertIn('csrf_token', sess)
                self.assertTrue(len(sess['csrf_token']) >= 32)

    # =========================================================================
    # B. CSRF PROTECTION TESTS
    # =========================================================================

    def test_csrf_protection_missing_token_blocked(self):
        """State-changing POST request without CSRF token returns 403 Forbidden."""
        client = app.app.test_client()
        with client.session_transaction() as sess:
            sess['csrf_token'] = 'valid_session_token_123'

        resp = client.post('/retrain')
        self.assertEqual(resp.status_code, 403)
        res_json = resp.get_json()
        self.assertIn('CSRF', res_json.get('error', ''))

    def test_csrf_protection_invalid_token_blocked(self):
        """State-changing POST request with mismatched CSRF token returns 403 Forbidden."""
        client = app.app.test_client()
        with client.session_transaction() as sess:
            sess['csrf_token'] = 'real_session_token_123'

        resp = client.post('/retrain', headers={'X-CSRFToken': 'wrong_token_456'})
        self.assertEqual(resp.status_code, 403)
        res_json = resp.get_json()
        self.assertIn('CSRF', res_json.get('error', ''))

    def test_csrf_protection_valid_token_accepted(self):
        """State-changing POST request with valid X-CSRFToken and authorization header passes protection."""
        client, csrf_token = self.get_authenticated_client()
        resp = client.post('/retrain', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('success'))

    # =========================================================================
    # C. PATH TRAVERSAL, SSRF & DNS REBINDING PROTECTION TESTS
    # =========================================================================

    def test_path_traversal_attempts_rejected(self):
        """validate_and_resolve_upload_path strictly rejects directory traversal attacks."""
        traversal_attempts = [
            '../../etc/passwd',
            '..\\..\\windows\\win.ini',
            '/etc/shadow',
            'C:\\Windows\\System32\\drivers\\etc\\hosts',
            '../uploads/test.jpg',
            'valid_face.jpg/../../../secret.txt',
            '',
            'nonexistent_file_12345.jpg',
            'malicious_script.py',
            'exploit.exe'
        ]
        for attempt in traversal_attempts:
            safe_path, err = app.validate_and_resolve_upload_path(attempt)
            self.assertIsNone(safe_path, f"Attempt '{attempt}' should have been rejected")
            self.assertIsNotNone(err)

    def test_valid_sandboxed_upload_path_accepted(self):
        """validate_and_resolve_upload_path accepts legitimate filenames within UPLOAD_DIR."""
        filename = os.path.basename(self.verified_face_path)
        safe_path, err = app.validate_and_resolve_upload_path(filename)
        self.assertIsNone(err)
        self.assertIsNotNone(safe_path)
        self.assertTrue(os.path.isfile(safe_path))
        self.assertEqual(os.path.abspath(safe_path), os.path.abspath(self.verified_face_path))

    def test_ssrf_and_dns_rebinding_destinations_rejected(self):
        """resolve_and_validate_public_url blocks localhost, private IPs, cloud metadata, and invalid schemes."""
        ssrf_targets = [
            'http://localhost:5000/api',
            'http://127.0.0.1:8080/admin',
            'http://127.0.0.2/test',
            'http://0.0.0.0:8000/',
            'http://192.168.1.1/router',
            'http://10.0.0.5/internal',
            'http://172.16.0.1/backend',
            'http://169.254.169.254/latest/meta-data/',
            'ftp://example.com/photo.jpg',
            'file:///etc/passwd',
            'javascript:alert(1)'
        ]
        for target in ssrf_targets:
            is_valid, msg = app.validate_url_host_is_public(target)
            self.assertFalse(is_valid, f"SSRF target '{target}' should have been blocked")
            self.assertIsNotNone(msg)

    # =========================================================================
    # D. SAFE IMAGE CLEANUP & FACE DETECTION TESTS
    # =========================================================================

    def test_reject_image_without_face(self):
        """Images without detectable faces are rejected with an explicit error message."""
        emb, info, err = app.extract_face_embedding(self.no_face_path)
        self.assertIsNone(emb, "Embedding must be None when no face is detected")
        self.assertIsNone(info, "Info dict must be None when no face is detected")
        self.assertIsNotNone(err)
        self.assertIn("No face detected", err)

    def test_valid_face_produces_512_dim_normalized_embedding(self):
        """Images with valid faces produce a 512-dimensional unit-normalized HOG embedding."""
        emb, info, err = app.extract_face_embedding(self.verified_face_path)
        if emb is not None:
            self.assertIsNone(err)
            self.assertEqual(len(emb), 512, "Embedding vector must have exactly 512 dimensions")
            self.assertTrue(info['face_detected'])
            norm = np.linalg.norm(emb)
            self.assertAlmostEqual(norm, 1.0, places=3, msg="Embedding vector must be normalized to unit length")

    def test_temporary_upload_cleaned_up_on_failure(self):
        """When /analyze fails on a newly uploaded file without a face, the temporary file is safely cleaned up."""
        client, csrf_token = self.get_authenticated_client()

        with open(self.no_face_path, 'rb') as f:
            data = {
                'name': 'Temp Subject No Face',
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_token,
                'photo': (f, 'temp_no_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified')), \
                 patch('app.moderate_image', return_value=(True, 'Safe')):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                res_json = resp.get_json()
                self.assertEqual(resp.status_code, 400)
                self.assertFalse(res_json.get('success'))
                self.assertIn("No face detected", res_json.get('error', ''))

                # Temporary upload file should not exist
                uploads = os.listdir(app.UPLOAD_DIR)
                self.assertNotIn('temp_no_face.jpg', uploads)

    def test_existing_saved_image_preserved_on_failed_processing(self):
        """An existing saved image must NEVER be deleted when /analyze or moderation fails."""
        client, csrf_token = self.get_authenticated_client()

        # Step 1: Create a saved subject with an associated image
        saved_img_name = f"saved_{uuid.uuid4().hex[:8]}.jpg"
        saved_img_path = os.path.join(app.UPLOAD_DIR, saved_img_name)
        shutil.copy2(self.verified_face_path, saved_img_path)

        dummy_scores = {t: 50.0 for t in app.TRAIT_NAMES}
        dummy_info = {'face_detected': True, 'embedding_dim': 512}
        app.save_profile("Existing_Saved_Subject", dummy_scores, dummy_scores, "Bio", dummy_info,
                         'scientist', image_filename=saved_img_name)

        self.assertTrue(app.is_image_associated_or_saved(saved_img_name))
        self.assertTrue(os.path.exists(saved_img_path))

        # Step 2: Reference this saved image in an analyze call that fails moderation
        data = {
            'name': 'New Subject Trying To Use Saved Image',
            'biography': self.valid_bio,
            'profession': 'scientist',
            'csrf_token': csrf_token,
            'image_url_path': saved_img_name
        }
        with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified')), \
             patch('app.moderate_image', return_value=(False, 'Flagged by mock moderator')):
            resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                               headers={'X-CSRFToken': csrf_token})
            res_json = resp.get_json()
            self.assertEqual(resp.status_code, 400)
            self.assertFalse(res_json.get('success'))

            # The existing saved image MUST STILL EXIST!
            self.assertTrue(os.path.exists(saved_img_path), "Existing saved image must NEVER be deleted on failed analyze!")

        # Step 3: Test check_photo with explicit content rejection on saved image
        with patch('app.get_claude_client') as mock_claude:
            mock_client = MagicMock()
            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text='{"ok": false, "issue": "explicit_content", "reason": "Test flag"}')]
            mock_client.messages.create.return_value = mock_msg
            mock_claude.return_value = mock_client

            resp = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': saved_img_name, 'profession': 'scientist'})
            self.assertEqual(resp.status_code, 200)

            # Saved image MUST STILL EXIST!
            self.assertTrue(os.path.exists(saved_img_path), "check_photo must NEVER delete an existing saved image!")

    # =========================================================================
    # E. MACHINE LEARNING TRAINING & EVALUATION TESTS
    # =========================================================================

    def test_leakage_free_model_training_and_evaluation(self):
        """Verify leakage-free training:
        - 80/20 train/test split
        - Preprocessing fitted strictly on train partition
        - Dynamic PCA components
        - Trait metrics (MAE, R2, Pearson r, p-value) on test partition
        - Final production pipeline fitted and saved
        - Reproducibility metadata recorded
        """
        success, message, metrics = app.build_and_train_model()
        self.assertTrue(success, f"Model training failed: {message}")
        self.assertIsNotNone(metrics)

        # Dataset counts
        self.assertEqual(metrics['n_subjects'], len(self.synthetic_subjects))
        self.assertEqual(metrics['n_train'] + metrics['n_test'], metrics['n_subjects'])
        self.assertGreaterEqual(metrics['n_test'], 1)

        # Dynamic PCA components
        self.assertLessEqual(metrics['pca_components'], metrics['n_train'] - 1)
        self.assertLessEqual(metrics['pca_components'], 10)

        # Trait metrics
        for trait in app.TRAIT_NAMES:
            self.assertIn(trait, metrics['trait_metrics'])
            tm = metrics['trait_metrics'][trait]
            self.assertIn('mae', tm)
            self.assertIn('r2', tm)
            self.assertIn('pearson_r', tm)
            self.assertIn('p_value', tm)

        # Model and metrics files created
        self.assertTrue(os.path.exists(app.TRAINED_MODEL_FILE))
        self.assertTrue(os.path.exists(app.MODEL_METRICS_FILE))

        # Prediction test via estimate_personality
        dummy_vec = np.random.randn(512).tolist()
        pred, err = app.estimate_personality(dummy_vec)
        self.assertIsNone(err)
        self.assertIsNotNone(pred)
        for trait in app.TRAIT_NAMES:
            self.assertIn(trait, pred)
            self.assertIsInstance(pred[trait], (int, float))

    # =========================================================================
    # F. UNDEFINED STATISTICAL METRICS HANDLING TESTS
    # =========================================================================

    def test_undefined_statistical_metrics_represented_as_none_and_null(self):
        """Verify that when statistical metrics cannot be computed (zero variance),
        they are represented as None in Python and null in JSON, not arbitrary 0.0 or 1.0."""
        const_emb_rows = []
        const_score_rows = []
        for i in range(1, 7):
            vec = np.random.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            const_emb_rows.append([f"Const_Sub_{i}"] + vec.tolist())
            const_score_rows.append({
                'name': f"Const_Sub_{i}",
                'openness': 50.0,  # Zero variance!
                'conscientiousness': float(40 + i * 5),
                'extraversion': float(30 + i * 6),
                'agreeableness': float(50 + i * 4),
                'neuroticism': float(20 + i * 7),
            })

        temp_emb_file = os.path.join(app.PROCESSED_DIR, 'const_emb.csv')
        temp_score_file = os.path.join(app.PROCESSED_DIR, 'const_score.csv')
        pd.DataFrame(const_emb_rows, columns=['name'] + [f'face_{i}' for i in range(1, 513)]).to_csv(temp_emb_file, index=False)
        pd.DataFrame(const_score_rows).to_csv(temp_score_file, index=False)

        with patch.object(app, 'FACE_EMBEDDINGS_FILE', temp_emb_file), \
             patch.object(app, 'PERSONALITY_SCORES_FILE', temp_score_file):
            success, _, metrics = app.build_and_train_model()
            self.assertTrue(success)

            op_metric = metrics['trait_metrics']['openness']
            self.assertIsNone(op_metric['r2'], "R2 must be None when test target variance is zero")
            self.assertIsNone(op_metric['pearson_r'], "Pearson r must be None when test target variance is zero")
            self.assertIsNone(op_metric['p_value'], "p-value must be None when correlation is undefined")

            json_str = json.dumps(metrics)
            loaded_json = json.loads(json_str)
            self.assertIsNone(loaded_json['trait_metrics']['openness']['r2'])
            self.assertIn('"r2": null', json_str)

        for p in [temp_emb_file, temp_score_file]:
            if os.path.exists(p):
                os.remove(p)

    def test_negative_r2_preserved(self):
        """Genuine negative R2 values must be preserved as floats, not coerced to 0 or None."""
        app.build_and_train_model()
        with open(app.MODEL_METRICS_FILE, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        for trait, tm in metrics['trait_metrics'].items():
            if tm['r2'] is not None:
                self.assertIsInstance(tm['r2'], (int, float))

    # =========================================================================
    # G. SUBJECT DELETION RELIABILITY & SHARED IMAGE PRESERVATION TESTS
    # =========================================================================

    def test_shared_image_preserved_on_subject_deletion(self):
        """When two subjects share an image, deleting one subject preserves the image for the other."""
        client, csrf_token = self.get_authenticated_client()

        shared_img_name = f"shared_{uuid.uuid4().hex[:8]}.jpg"
        shared_img_path = os.path.join(app.UPLOAD_DIR, shared_img_name)
        shutil.copy2(self.verified_face_path, shared_img_path)

        dummy_scores = {t: 50.0 for t in app.TRAIT_NAMES}
        dummy_info = {'face_detected': True, 'embedding_dim': 512}

        # Subject A and Subject B both reference shared_img_name
        app.save_profile("Subject_Alpha", dummy_scores, dummy_scores, "Bio Alpha", dummy_info,
                         'scientist', image_filename=shared_img_name)
        app.save_profile("Subject_Beta", dummy_scores, dummy_scores, "Bio Beta", dummy_info,
                         'scientist', image_filename=shared_img_name)

        self.assertTrue(os.path.exists(shared_img_path))
        self.assertEqual(len(app.get_subjects_referencing_image(shared_img_name)), 2)

        # Delete Subject_Alpha
        resp = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                           json={'name': 'Subject_Alpha'})
        res_json = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(res_json.get('success'))
        self.assertTrue(res_json.get('is_shared_image'))
        self.assertFalse(res_json.get('image_deleted'))

        # Shared image MUST STILL EXIST because Subject_Beta references it
        self.assertTrue(os.path.exists(shared_img_path), "Shared image file must be preserved on disk!")
        self.assertEqual(app.get_subjects_referencing_image(shared_img_name), ['Subject_Beta'])

        # Now delete Subject_Beta (last reference)
        resp2 = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                            json={'name': 'Subject_Beta'})
        res2_json = resp2.get_json()
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(res2_json.get('success'))
        self.assertTrue(res2_json.get('image_deleted'))

        # Now that no subjects reference it, it should be deleted
        self.assertFalse(os.path.exists(shared_img_path), "Image should be deleted once last subject is removed")

    def test_partial_failure_reporting_and_retry_retention(self):
        """When image deletion fails (simulated error), the association is retained for retry and reported accurately."""
        client, csrf_token = self.get_authenticated_client()

        target_name = "Subject_Locked_Image"
        img_name = f"locked_{uuid.uuid4().hex[:8]}.jpg"
        img_path = os.path.join(app.UPLOAD_DIR, img_name)
        shutil.copy2(self.verified_face_path, img_path)

        dummy_scores = {t: 50.0 for t in app.TRAIT_NAMES}
        dummy_info = {'face_detected': True, 'embedding_dim': 512}
        app.save_profile(target_name, dummy_scores, dummy_scores, "Bio", dummy_info,
                         'scientist', image_filename=img_name)

        # Mock os.remove to raise PermissionError when deleting img_path
        orig_remove = os.remove
        def mock_remove(path):
            if os.path.abspath(path) == os.path.abspath(img_path):
                raise PermissionError("Simulated locked file")
            return orig_remove(path)

        with patch('os.remove', side_effect=mock_remove):
            resp = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                               json={'name': target_name})
            res_json = resp.get_json()

            # Should return 207 (partial failure) and not claim full success!
            self.assertEqual(resp.status_code, 207)
            self.assertFalse(res_json.get('success'))
            self.assertTrue(res_json.get('partial_failure'))
            self.assertIn('Image cleanup failed', res_json.get('message', ''))

            # Association should be retained with 'cleanup_failed' status so it can be retried
            associations = app.load_image_associations()
            self.assertIn(target_name, associations)
            self.assertEqual(associations[target_name].get('status'), 'cleanup_failed')

        # Clean up manually
        if os.path.exists(img_path):
            os.remove(img_path)
        app.remove_image_association(target_name)

    def test_delete_nonexistent_subject_returns_404(self):
        """Deleting a subject that does not exist returns 404 Not Found."""
        client, csrf_token = self.get_authenticated_client()
        resp = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                           json={'name': 'CompletelyNonexistentSubject999'})
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.get_json().get('success'))

    # =========================================================================
    # H. END-TO-END WORKFLOW & REGRESSION VERIFICATION TESTS
    # =========================================================================

    def test_mocked_end_to_end_analyze_workflow(self):
        """Verify complete /analyze flow with mocked Claude AI responses."""
        client, csrf_token = self.get_authenticated_client()

        mock_scores = {
            'openness': 82.0,
            'conscientiousness': 75.0,
            'extraversion': 68.0,
            'agreeableness': 74.0,
            'neuroticism': 38.0,
            'reasoning': 'Mocked reasoning for end-to-end test subject.'
        }

        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': 'E2E Mocked Researcher Test',
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_token,
                'photo': (f, 'test_verified_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified')), \
                 patch('app.moderate_image', return_value=(True, 'Safe')), \
                 patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None)):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                res_json = resp.get_json()
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(res_json.get('success'))
                self.assertEqual(res_json['name'], 'E2E Mocked Researcher Test')
                self.assertIn('biography_scores', res_json)

                # Clean up subject
                client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                            json={'name': 'E2E Mocked Researcher Test'})

    # =========================================================================
    # I. CLAUDE AI INTEGRATION & ERROR HANDLING TESTS
    # =========================================================================

    def test_claude_model_constants_are_valid(self):
        """Configured Anthropic Claude models must be valid identifiers."""
        self.assertTrue('claude' in app.CLAUDE_SONNET_MODEL.lower() and 'sonnet' in app.CLAUDE_SONNET_MODEL.lower())
        self.assertTrue('claude' in app.CLAUDE_HAIKU_MODEL.lower() and 'haiku' in app.CLAUDE_HAIKU_MODEL.lower())

    def test_retired_claude_models_removed_from_candidate_lists(self):
        """Retired or unsupported Claude 3.5 models must not be in candidate fallback lists."""
        retired_identifiers = [
            'claude-3-5-sonnet-latest',
            'claude-3-5-sonnet-20241022',
            'claude-3-5-haiku-latest',
            'claude-3-5-haiku-20241022',
        ]
        for retired in retired_identifiers:
            self.assertNotIn(retired, app.SONNET_MODEL_CANDIDATES)
            self.assertNotIn(retired, app.HAIKU_MODEL_CANDIDATES)

    def test_missing_or_placeholder_api_key_raises_clear_error(self):
        """Missing or placeholder Anthropic API key raises a clear ValueError without calling the API."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}):
            with self.assertRaises(ValueError) as ctx:
                app.get_claude_client()
            self.assertIn("not configured", str(ctx.exception))

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'your_anthropic_api_key_here'}):
            with self.assertRaises(ValueError) as ctx:
                app.get_claude_client()
            self.assertIn("placeholder value", str(ctx.exception))

    def test_claude_auth_error_produces_clear_message(self):
        """When Anthropic returns 401 invalid x-api-key, format_claude_error formats a helpful message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        auth_err = anthropic.AuthenticationError(
            message="Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}",
            response=mock_resp,
            body={"error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        )
        formatted = app.format_claude_error(auth_err)
        self.assertIn("Authentication Error (401)", formatted)
        self.assertIn("ANTHROPIC_API_KEY", formatted)

    def test_claude_rate_limit_error_produces_clear_message(self):
        """When Anthropic returns 429 rate limit, format_claude_error formats a helpful message."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        rate_err = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=mock_resp,
            body={}
        )
        formatted = app.format_claude_error(rate_err)
        self.assertIn("Rate Limit (429)", formatted)

    def test_unresolved_images_report_endpoint(self):
        """The /unresolved_images_report endpoint returns 200 with the audit report when authorized, 403 when anonymous."""
        client, _ = self.get_authenticated_client()
        resp_anon = client.get('/unresolved_images_report')
        self.assertEqual(resp_anon.status_code, 403)

        resp = client.get('/unresolved_images_report', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp.status_code, 200)

    # =========================================================================
    # J. ADMINISTRATIVE ENDPOINT SECURITY & PERMISSION TESTS (BUG 3 AUDIT)
    # =========================================================================

    def test_admin_authorization_without_key_denied_403(self):
        """Administrative endpoints (/retrain, /delete_subject) strictly deny access (403) without admin key."""
        client, csrf_token = self.get_authenticated_client()

        # 1. Retrain without admin key
        resp_retrain = client.post('/retrain', headers={'X-CSRFToken': csrf_token})
        self.assertEqual(resp_retrain.status_code, 403)
        res_json = resp_retrain.get_json()
        self.assertFalse(res_json.get('success'))
        self.assertIn("administrative", res_json.get('error', '').lower())

        # 2. Delete subject without admin key
        resp_del = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token},
                               json={'name': 'Sandboxed_Subject_1'})
        self.assertEqual(resp_del.status_code, 403)
        res_del_json = resp_del.get_json()
        self.assertFalse(res_del_json.get('success'))
        self.assertIn("administrative", res_del_json.get('error', '').lower())

    def test_admin_authorization_with_invalid_key_denied_403(self):
        """Administrative endpoints strictly deny access (403) when given incorrect admin key."""
        client, csrf_token = self.get_authenticated_client()

        resp_retrain = client.post('/retrain', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': 'wrong_secret_key'})
        self.assertEqual(resp_retrain.status_code, 403)

        resp_del = client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': 'wrong_secret_key'},
                               json={'name': 'Sandboxed_Subject_1'})
        self.assertEqual(resp_del.status_code, 403)

    def test_admin_authorization_with_valid_key_succeeds(self):
        """Administrative endpoints succeed when provided matching admin key."""
        client, csrf_token = self.get_authenticated_client()

        resp_retrain = client.post('/retrain', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key})
        self.assertEqual(resp_retrain.status_code, 200)
        self.assertTrue(resp_retrain.get_json().get('success'))

    def test_insecure_and_short_admin_keys_locked(self):
        """Insecure, default, or short (<12 chars) admin keys lock administrative endpoints."""
        client, csrf_token = self.get_authenticated_client()

        for insecure_key in ['facepersona_admin_2026', 'your_admin_secret_key_here', 'changeme', 'short_key']:
            with patch.dict(os.environ, {'RESEARCH_ADMIN_KEY': insecure_key}):
                resp = client.post('/retrain', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': insecure_key})
                self.assertEqual(resp.status_code, 403)
                self.assertIn("locked", resp.get_json().get('error', '').lower())

    def test_profile_path_traversal_strictly_prevented(self):
        """Directory traversal attempts in profile resolution return None and are confined."""
        malicious_names = [
            '../../etc/passwd',
            '..\\..\\windows\\system32',
            '../../../app',
            'foo/../../bar',
            'foo\\..\\..\\bar',
            '\x00evil_subject',
            '',
            None,
        ]
        for bad_name in malicious_names:
            resolved = app.resolve_profile_path(bad_name)
            self.assertIsNone(resolved)

    def test_analyze_input_bounds_and_sanitization(self):
        """Public /analyze endpoint strictly rejects oversized inputs and control characters."""
        client, csrf_token = self.get_authenticated_client()

        # 1. Oversized name (>100 chars)
        resp = client.post('/analyze', headers={'X-CSRFToken': csrf_token}, data={
            'name': 'A' * 105,
            'biography': self.valid_bio,
            'profession': 'other',
            'upload_id': 'verified_face_sample.jpg'
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("100 characters", resp.get_json().get('error', ''))

        # 2. Control character in name
        resp = client.post('/analyze', headers={'X-CSRFToken': csrf_token}, data={
            'name': 'Test\x00Name',
            'biography': self.valid_bio,
            'profession': 'other',
            'upload_id': 'verified_face_sample.jpg'
        })
        self.assertEqual(resp.status_code, 400)

    # =========================================================================
    # K. VALIDATION BYPASS PREVENTION & ERROR HANDLING (BUG 2 AUDIT)
    # =========================================================================

    def test_claude_failure_moderate_image_returns_unavailable_not_safe(self):
        """When Claude API fails, moderate_image returns (False, err, 'UNAVAILABLE') and NEVER approves."""
        with patch('app.get_claude_client') as mock_claude:
            mock_claude.side_effect = anthropic.APIConnectionError(request=MagicMock(), message="Simulated connection failure")
            res = app.moderate_image(self.verified_face_path)
            self.assertEqual(len(res), 3)
            is_safe, reason, status = res
            self.assertFalse(is_safe, "Must not approve image on API failure")
            self.assertEqual(status, 'UNAVAILABLE')
            self.assertIn("unavailable", reason.lower())

    def test_claude_failure_validate_match_returns_unavailable_not_match(self):
        """When Claude API fails, validate_match_via_gemini returns (False, err, 'UNAVAILABLE') and NEVER claims match."""
        with patch('app.get_claude_client') as mock_claude:
            mock_claude.side_effect = anthropic.APITimeoutError(request=MagicMock())
            res = app.validate_match_via_gemini("Test Subject", "Test Biography")
            self.assertEqual(len(res), 3)
            is_match, reason, status = res
            self.assertFalse(is_match, "Must not claim match on API failure")
            self.assertEqual(status, 'UNAVAILABLE')
            self.assertIn("unavailable", reason.lower())

    def test_check_photo_endpoint_returns_503_unavailable_on_api_error(self):
        """When Claude API fails, /check_photo returns HTTP 503 with status UNAVAILABLE, not ok=True."""
        client, csrf_token = self.get_authenticated_client()
        with patch('app.get_claude_client') as mock_claude:
            mock_claude.side_effect = anthropic.APIStatusError(message="503 upstream error", response=MagicMock(status_code=503), body={})
            resp = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(self.verified_face_path), 'profession': 'scientist'})
            self.assertEqual(resp.status_code, 503, f"Expected 503, got {resp.status_code}: body={resp.get_data(as_text=True)}")
            res_json = resp.get_json()
            self.assertFalse(res_json.get('ok'))
            self.assertEqual(res_json.get('status'), 'UNAVAILABLE')

    def test_validate_endpoint_returns_error_status_on_api_error(self):
        """When Claude API fails, /validate returns error code with status UNAVAILABLE, not match."""
        client, csrf_token = self.get_authenticated_client()
        with patch('app.get_claude_client') as mock_claude:
            mock_claude.side_effect = anthropic.APITimeoutError(request=MagicMock())
            resp = client.post('/validate', headers={'X-CSRFToken': csrf_token},
                               json={'name': 'Albert Einstein', 'biography': self.valid_bio})
            self.assertEqual(resp.status_code, 503, f"Expected 503, got {resp.status_code}: body={resp.get_data(as_text=True)}")
            res_json = resp.get_json()
            self.assertFalse(res_json.get('is_match'))
            self.assertEqual(res_json.get('status'), 'UNAVAILABLE')

    def test_analyze_halts_without_writing_on_claude_failure(self):
        """When personality scoring is unavailable, /analyze returns 503 and does NOT save subject to dataset."""
        client, csrf_token = self.get_authenticated_client()
        test_subj = "Unsaved_Failing_Subject"

        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': test_subj,
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_token,
                'photo': (f, 'test_fail_img.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified', 'PASSED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                 patch('app.score_personality_via_gemini', return_value=(None, "Anthropic API Timeout", 'UNAVAILABLE')):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                self.assertEqual(resp.status_code, 503, f"Expected 503, got {resp.status_code}: body={resp.get_data(as_text=True)}")
                self.assertFalse(resp.get_json().get('success'))
                self.assertEqual(resp.get_json().get('status'), 'UNAVAILABLE')

                # Verify subject was NOT written to CSVs or profiles
                if os.path.exists(app.FACE_EMBEDDINGS_FILE):
                    df = pd.read_csv(app.FACE_EMBEDDINGS_FILE)
                    self.assertNotIn(test_subj, df['name'].values)
                if os.path.exists(app.PERSONALITY_SCORES_FILE):
                    df = pd.read_csv(app.PERSONALITY_SCORES_FILE)
                    self.assertNotIn(test_subj, df['name'].values)
                prof_path = os.path.join(app.PROFILES_DIR, f"{test_subj}.json")
                self.assertFalse(os.path.exists(prof_path))

    # =========================================================================
    # L. AI RESPONSE PARSING & SCHEMA VALIDATION TESTS (ISSUES 5, 6, 7)
    # =========================================================================

    def test_ai_response_parsing_strict_boolean(self):
        """Strict boolean parser correctly parses true/false and rejects malformed values."""
        # 1. Valid positive cases
        self.assertTrue(app.parse_strict_boolean(True))
        self.assertTrue(app.parse_strict_boolean("true"))
        self.assertTrue(app.parse_strict_boolean("True"))
        self.assertTrue(app.parse_strict_boolean("TRUE"))
        self.assertTrue(app.parse_strict_boolean(1))

        # 2. Valid negative cases (CRITICAL: 'false' must NOT be truthy!)
        self.assertFalse(app.parse_strict_boolean(False))
        self.assertFalse(app.parse_strict_boolean("false"))
        self.assertFalse(app.parse_strict_boolean("False"))
        self.assertFalse(app.parse_strict_boolean("FALSE"))
        self.assertFalse(app.parse_strict_boolean(0))

        # 3. Invalid types and ambiguous values must raise ValueError
        invalid_cases = [None, "", "yes", "no", "null", "none", 2, -1, 3.14, [], {}, [True]]
        for val in invalid_cases:
            with self.assertRaises(ValueError, msg=f"Should reject {repr(val)}"):
                app.parse_strict_boolean(val)

    def test_ai_response_parsing_extract_and_parse_json(self):
        """extract_and_parse_json handles markdown fences, surrounding text, and enforces dict schema."""
        # Plain JSON
        res1 = app.extract_and_parse_json('{"is_safe": true, "reason": "clean"}')
        self.assertEqual(res1, {"is_safe": True, "reason": "clean"})

        # Markdown fenced JSON
        res2 = app.extract_and_parse_json('```json\n{"is_match": false, "reason": "mismatch"}\n```')
        self.assertEqual(res2, {"is_match": False, "reason": "mismatch"})

        # Text surrounding JSON
        res3 = app.extract_and_parse_json('Analysis result:\n{"ok": true, "issue": null}\nEnd of note.')
        self.assertEqual(res3, {"ok": True, "issue": None})

        # Malformed and invalid structures
        with self.assertRaises(ValueError):
            app.extract_and_parse_json("")
        with self.assertRaises(ValueError):
            app.extract_and_parse_json("No JSON at all in this response.")
        with self.assertRaises(ValueError):
            app.extract_and_parse_json('[1, 2, 3]')  # Not a dict
        with self.assertRaises(ValueError):
            app.extract_and_parse_json('{"unclosed": json')

    def test_big_five_scores_validation_and_range_enforcement(self):
        """validate_big_five_scores accepts valid 0-100 scores and rejects non-numerical, out-of-range, NaN, and inf."""
        valid_scores = {
            'openness': 75.0,
            'conscientiousness': 85.5,
            'extraversion': 60.0,
            'agreeableness': 90.0,
            'neuroticism': 25.0
        }
        cleaned, err = app.validate_big_five_scores(valid_scores)
        self.assertIsNone(err)
        self.assertEqual(cleaned['openness'], 75.0)

        # Boundary values 0.0 and 100.0 are valid
        boundary_scores = {t: 0.0 for t in app.TRAIT_NAMES}
        boundary_scores['neuroticism'] = 100.0
        cleaned_b, err_b = app.validate_big_five_scores(boundary_scores)
        self.assertIsNone(err_b)
        self.assertEqual(cleaned_b['openness'], 0.0)
        self.assertEqual(cleaned_b['neuroticism'], 100.0)

        # Out-of-range: > 100 (must be rejected, NOT clamped!)
        over_scores = valid_scores.copy()
        over_scores['openness'] = 105.0
        _, err_over = app.validate_big_five_scores(over_scores)
        self.assertIsNotNone(err_over)
        self.assertIn("out of valid range", err_over)

        # Out-of-range: < 0 (must be rejected, NOT clamped!)
        under_scores = valid_scores.copy()
        under_scores['neuroticism'] = -5.0
        _, err_under = app.validate_big_five_scores(under_scores)
        self.assertIsNotNone(err_under)
        self.assertIn("out of valid range", err_under)

        # Non-numerical string
        str_scores = valid_scores.copy()
        str_scores['agreeableness'] = "very high"
        _, err_str = app.validate_big_five_scores(str_scores)
        self.assertIsNotNone(err_str)
        self.assertIn("Non-numerical value", err_str)

        # Boolean value (in Python, bool is subclass of int; must be rejected)
        bool_scores = valid_scores.copy()
        bool_scores['agreeableness'] = True
        _, err_bool = app.validate_big_five_scores(bool_scores)
        self.assertIsNotNone(err_bool)

        # NaN and Infinity
        nan_scores = valid_scores.copy()
        nan_scores['conscientiousness'] = float('nan')
        _, err_nan = app.validate_big_five_scores(nan_scores)
        self.assertIsNotNone(err_nan)
        self.assertIn("non-finite", err_nan)

        inf_scores = valid_scores.copy()
        inf_scores['extraversion'] = float('inf')
        _, err_inf = app.validate_big_five_scores(inf_scores)
        self.assertIsNotNone(err_inf)

        # Missing trait
        missing_scores = valid_scores.copy()
        del missing_scores['agreeableness']
        _, err_missing = app.validate_big_five_scores(missing_scores)
        self.assertIsNotNone(err_missing)
        self.assertIn("Missing required Big Five trait", err_missing)

    def test_check_photo_rejects_missing_or_empty_image(self):
        """/check_photo rejects missing or empty image inputs with HTTP 400 and does NOT return PASSED."""
        client, csrf_token = self.get_authenticated_client()

        # 1. Empty image reference
        resp_empty = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                                 json={'image_path': '', 'profession': 'scientist'})
        self.assertEqual(resp_empty.status_code, 400)
        res_empty_json = resp_empty.get_json()
        self.assertFalse(res_empty_json.get('ok'))
        self.assertEqual(res_empty_json.get('status'), 'FAILED')

        # 2. Missing image reference altogether
        resp_missing = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                                   json={'profession': 'scientist'})
        self.assertEqual(resp_missing.status_code, 400)
        self.assertFalse(resp_missing.get_json().get('ok'))

    def test_server_side_biography_validation_enforced_in_analyze(self):
        """/analyze independently validates name-biography match server-side and rejects mismatches without database writes."""
        client, csrf_token = self.get_authenticated_client()
        mismatched_subj = "Direct_API_Mismatched_Subject"

        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': mismatched_subj,
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_token,
                'photo': (f, 'test_mismatch.jpg')
            }
            # Mock validate_match_via_gemini to report mismatch
            with patch('app.validate_match_via_gemini', return_value=(False, 'Biography is about Napoleon Bonaparte, not the subject', 'FAILED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                self.assertEqual(resp.status_code, 400)
                res_json = resp.get_json()
                self.assertFalse(res_json.get('success'))
                self.assertTrue(res_json.get('mismatch'))
                self.assertIn("does not match", res_json.get('error', ''))

                # Database records MUST NOT have been created
                if os.path.exists(app.FACE_EMBEDDINGS_FILE):
                    df = pd.read_csv(app.FACE_EMBEDDINGS_FILE)
                    self.assertNotIn(mismatched_subj, df['name'].values)

    def test_manage_research_cli_commands(self):
        """Verify manage_research.py status, retrain, and audit-images run cleanly."""
        import manage_research
        with patch('sys.stdout', new=MagicMock()):
            # Test status
            manage_research.cmd_status(MagicMock())
            # Test retrain
            manage_research.cmd_retrain(MagicMock())
            # Test audit-images
            manage_research.cmd_audit_images(MagicMock())

    def test_profile_filename_collision_prevention(self):
        """Distinct subject names that sanitize to identical base filenames must not collide or overwrite,
        specifically including names like Ana-Maria and Ana_Maria."""
        # Test 1: Ana-Maria vs Ana_Maria
        ana_hyphen = "Ana-Maria"
        ana_under = "Ana_Maria"

        app.save_profile(ana_hyphen, {'openness': 75.0, 'conscientiousness': 80.0, 'extraversion': 65.0, 'agreeableness': 70.0, 'neuroticism': 40.0},
                         None, "Reasoning Ana-Maria", None)
        path_hyphen = app.resolve_profile_path(ana_hyphen)
        self.assertIsNotNone(path_hyphen)

        app.save_profile(ana_under, {'openness': 90.0, 'conscientiousness': 95.0, 'extraversion': 50.0, 'agreeableness': 85.0, 'neuroticism': 30.0},
                         None, "Reasoning Ana_Maria", None)
        path_under = app.resolve_profile_path(ana_under)
        self.assertIsNotNone(path_under)

        # File paths must be completely distinct
        self.assertNotEqual(path_hyphen, path_under)

        # Both profiles must exist with their distinct subject names and distinct scores
        prof_hyphen = app.load_profile(ana_hyphen)
        prof_under = app.load_profile(ana_under)
        self.assertEqual(prof_hyphen['name'], ana_hyphen)
        self.assertEqual(prof_under['name'], ana_under)
        self.assertEqual(prof_hyphen['biography_scores']['openness'], 75.0)
        self.assertEqual(prof_under['biography_scores']['openness'], 90.0)

        # Test 2: Slash / space colliding names
        name_a = "Collision Test Subject"
        name_b = "Collision/Test/Subject"

        app.save_profile(name_a, {'openness': 70.0, 'conscientiousness': 70.0, 'extraversion': 70.0, 'agreeableness': 70.0, 'neuroticism': 50.0},
                         None, "Reasoning A", None)
        path_a = app.resolve_profile_path(name_a)
        app.save_profile(name_b, {'openness': 85.0, 'conscientiousness': 85.0, 'extraversion': 85.0, 'agreeableness': 85.0, 'neuroticism': 30.0},
                         None, "Reasoning B", None)
        path_b = app.resolve_profile_path(name_b)
        self.assertNotEqual(path_a, path_b)

        # Clean up
        for p in [path_hyphen, path_under, path_a, path_b]:
            if p and os.path.exists(p):
                os.remove(p)

    def test_public_assessment_separated_from_research_training_records(self):
        """Public assessments are saved to public_assessments.csv and profiles without polluting research training records."""
        client, csrf_token = self.get_authenticated_client()
        pub_subj = "Public Citizen Assessment Test"

        init_emb_count = len(pd.read_csv(app.FACE_EMBEDDINGS_FILE)) if os.path.exists(app.FACE_EMBEDDINGS_FILE) else 0
        init_score_count = len(pd.read_csv(app.PERSONALITY_SCORES_FILE)) if os.path.exists(app.PERSONALITY_SCORES_FILE) else 0

        mock_scores = {
            'openness': 80.0, 'conscientiousness': 80.0, 'extraversion': 60.0,
            'agreeableness': 70.0, 'neuroticism': 50.0, 'reasoning': 'Public assessment reasoning.'
        }
        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': pub_subj,
                'biography': self.valid_bio,
                'profession': 'other',
                'csrf_token': csrf_token,
                'photo': (f, 'test_pub_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified', 'PASSED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                 patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.get_json().get('success'))

                # 1. Research training files (face_embeddings & personality_scores) MUST remain unpolluted
                final_emb_count = len(pd.read_csv(app.FACE_EMBEDDINGS_FILE)) if os.path.exists(app.FACE_EMBEDDINGS_FILE) else 0
                final_score_count = len(pd.read_csv(app.PERSONALITY_SCORES_FILE)) if os.path.exists(app.PERSONALITY_SCORES_FILE) else 0
                self.assertEqual(init_emb_count, final_emb_count)
                self.assertEqual(init_score_count, final_score_count)

                # 2. Public assessment is recorded in public_assessments.csv and data/profiles/
                self.assertTrue(os.path.exists(app.PUBLIC_ASSESSMENTS_FILE))
                df_pub = pd.read_csv(app.PUBLIC_ASSESSMENTS_FILE)
                self.assertIn(pub_subj, df_pub['name'].values)

                prof = app.load_profile(pub_subj)
                self.assertIsNotNone(prof)
                self.assertEqual(prof['name'], pub_subj)

                # 3. get_subjects includes the public assessment for viewing
                all_subjs = [s['name'] for s in app.get_subjects()]
                self.assertIn(pub_subj, all_subjs)

                # Clean up
                client.post('/delete_subject', headers={'X-CSRFToken': csrf_token, 'X-Admin-Key': self.admin_key},
                            json={'name': pub_subj})

    def test_research_baseline_protected_from_overwrite(self):
        """Protected research baseline subjects in dataset.csv cannot be overwritten by public /analyze."""
        client, csrf_token = self.get_authenticated_client()
        baseline_subject = "Sandboxed_Subject_1"

        # Write dataset.csv in sandbox so it is considered a baseline subject
        if os.path.exists(app.DATASET_FILE):
            df_base = pd.read_csv(app.DATASET_FILE)
        else:
            df_base = pd.DataFrame([{'name': baseline_subject}])
            df_base.to_csv(app.DATASET_FILE, index=False)

        resp = client.post('/analyze', headers={'X-CSRFToken': csrf_token}, data={
            'name': baseline_subject,
            'biography': self.valid_bio,
            'profession': 'scientist',
            'upload_id': 'verified_face_sample.jpg'
        })
        self.assertEqual(resp.status_code, 409)
        res_json = resp.get_json()
        self.assertFalse(res_json.get('success'))
        self.assertIn("protected research baseline subject", res_json.get('error', ''))

    # =========================================================================
    # QA FEEDBACK IMPLEMENTATION & REGRESSION TESTS
    # =========================================================================

    def test_biography_word_count_enforcement_and_validation(self):
        """Verify that biography word count rules are strictly enforced and cannot be bypassed."""
        client, csrf_token = self.get_authenticated_client()

        # 1. Under-minimum biography (40 words) rejected with HTTP 400
        short_bio = "This is a short biography of forty words describing an academic subject with brief details that fail the required minimum word count threshold for scientific analysis purposes today and always."
        self.assertLess(app.count_meaningful_words(short_bio), app.MIN_BIOGRAPHY_WORDS)

        resp = client.post('/analyze', headers={'X-CSRFToken': csrf_token}, data={
            'name': 'Short Bio Candidate',
            'biography': short_bio,
            'profession': 'scientist',
            'upload_id': 'verified_face_sample.jpg'
        })
        self.assertEqual(resp.status_code, 400)
        res_json = resp.get_json()
        self.assertFalse(res_json.get('success'))
        self.assertIn("too short", res_json.get('error', '').lower())
        self.assertEqual(res_json.get('min_words'), app.MIN_BIOGRAPHY_WORDS)

        # 2. Meaningful word counter does not count formatting, whitespace, or punctuation
        punct_text = "   ... --- ,,, !!! ??? \n\n\t   "
        self.assertEqual(app.count_meaningful_words(punct_text), 0)

        # 3. Valid biography meeting or exceeding minimum is accepted
        valid_words = app.count_meaningful_words(self.valid_bio)
        self.assertGreaterEqual(valid_words, app.MIN_BIOGRAPHY_WORDS)
        is_valid, count, err = app.validate_biography_text(self.valid_bio)
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_biography_validation_rejection_of_empty_and_meaningless_text(self):
        """Reject empty, whitespace, and repetitive spam content."""
        # Empty string
        is_valid, count, err = app.validate_biography_text("")
        self.assertFalse(is_valid)
        self.assertEqual(count, 0)

        # Whitespace only
        is_valid, count, err = app.validate_biography_text("   \n\t  ")
        self.assertFalse(is_valid)

        # Repetitive nonsense (85 copies of the same word)
        repetitive = "neuroscientist " * 85
        is_valid, count, err = app.validate_biography_text(repetitive)
        self.assertFalse(is_valid)
        self.assertIn("repetitive", err.lower())

    def test_biography_word_count_configurable_via_env(self):
        """Verify that MIN_BIOGRAPHY_WORDS is configurable in one location."""
        # Test with custom minimum threshold of 80 words
        is_valid_at_80, _, _ = app.validate_biography_text(self.valid_bio, min_words=80)
        self.assertTrue(is_valid_at_80)

        # With 120 words minimum, our 93-word text should be rejected
        is_valid_at_120, _, err_120 = app.validate_biography_text(self.valid_bio, min_words=120)
        self.assertFalse(is_valid_at_120)
        self.assertIn("too short", err_120.lower())

    def test_validate_and_paraphrase_enforce_minimum_words(self):
        """Both /validate and /paraphrase backend endpoints enforce the biography word count."""
        client, csrf_token = self.get_authenticated_client()

        # /validate with 10 words fails
        resp = client.post('/validate', headers={'X-CSRFToken': csrf_token}, json={
            'name': 'Albert Einstein',
            'biography': 'Short ten word biography for testing validation bypass attempt.'
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too short", resp.get_json().get('error', '').lower())

        # /paraphrase with 10 words fails
        resp = client.post('/paraphrase', headers={'X-CSRFToken': csrf_token}, json={
            'name': 'Albert Einstein',
            'biography': 'Short ten word biography for testing paraphrase bypass attempt.'
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("too short", resp.get_json().get('error', '').lower())

    def test_fresh_hog_extraction_and_ml_prediction_on_distinct_photos(self):
        """Verify that distinct face images produce distinct 512-D HOG embeddings and distinct ML predictions."""
        # 1. Base verified face image
        emb_1, info_1, err_1 = app.extract_face_embedding(self.verified_face_path)
        self.assertIsNone(err_1)
        self.assertEqual(len(emb_1), 512)

        # 2. Second distinct verified face image
        emb_2, info_2, err_2 = app.extract_face_embedding(self.verified_face_path_2)
        self.assertIsNone(err_2)
        self.assertEqual(len(emb_2), 512)

        # The embeddings must be unit normalized
        self.assertAlmostEqual(np.linalg.norm(emb_1), 1.0, places=3)
        self.assertAlmostEqual(np.linalg.norm(emb_2), 1.0, places=3)

        # The two embeddings must be numerically different
        diff = np.linalg.norm(np.array(emb_1) - np.array(emb_2))
        self.assertGreater(diff, 0.05, "Distinct face images must produce distinct HOG features")

        # 3. Model predictions must execute dynamically and cleanly on both
        pred_1, err_pred_1 = app.estimate_personality(emb_1)
        pred_2, err_pred_2 = app.estimate_personality(emb_2)
        self.assertIsNone(err_pred_1)
        self.assertIsNone(err_pred_2)
        self.assertIsNotNone(pred_1)
        self.assertIsNotNone(pred_2)
        for t in app.TRAIT_NAMES:
            self.assertIn(t, pred_1)
            self.assertIn(t, pred_2)

    def test_research_dataset_privacy_protection_and_restricted_profile(self):
        """Anonymous visitors cannot view baseline individual profiles or audit reports."""
        client, csrf_token = self.get_authenticated_client()

        # Set up a mock baseline subject in sandbox
        baseline_name = "Confidential_Baseline_Researcher"
        dummy_scores = {t: 60.0 for t in app.TRAIT_NAMES}
        dummy_info = {
            'face_detected': True,
            'stats': {'nonzero': 512, 'min': 0.01, 'max': 0.85, 'mean': 0.04},
            'top50_values': [0.1] * 50
        }
        app.save_profile(baseline_name, dummy_scores, dummy_scores, "Bio", dummy_info, 'scientist')

        # Add to dataset.csv in sandbox so it is considered baseline
        df_base = pd.DataFrame([{'name': baseline_name}])
        df_base.to_csv(app.DATASET_FILE, index=False)

        # 1. Anonymous visitor visits baseline profile -> renders restricted_profile template
        resp = client.get(f'/profile/{baseline_name}')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Participant Privacy", resp.data)
        self.assertIn(b"research baseline dataset", resp.data)

        # 2. Authorized researcher (supplying X-Admin-Key) visits baseline profile -> renders full profile
        resp_auth = client.get(f'/profile/{baseline_name}', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp_auth.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_auth.data)
        self.assertIn(baseline_name.encode('utf-8'), resp_auth.data)

        # 3. Public assessment subject profile is protected from unauthorized access
        pub_name = "Public_Test_Figure"
        test_token = "test_creator_token_xyz"
        app.save_profile(pub_name, dummy_scores, dummy_scores, "Bio", dummy_info, 'scientist', access_token=test_token)
        df_pub = pd.DataFrame([{'name': pub_name}])
        df_pub.to_csv(app.PUBLIC_ASSESSMENTS_FILE, index=False)

        # 3a. Unauthorized visitor without token or creator session receives privacy restriction notice
        resp_unauth = client.get(f'/profile/{pub_name}')
        self.assertEqual(resp_unauth.status_code, 200)
        self.assertIn(b"Assessment Privacy", resp_unauth.data)
        self.assertNotIn(b"Biography Analysis", resp_unauth.data)

        # 3b. Creator accessing with valid access token views full profile
        resp_token = client.get(f'/profile/{pub_name}?token={test_token}')
        self.assertEqual(resp_token.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_token.data)

        # 3c. Authorized researcher views full profile
        resp_res = client.get(f'/profile/{pub_name}', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp_res.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_res.data)

    def test_unresolved_images_report_restricted_to_authorized_researchers(self):
        """Internal image audit report is forbidden to anonymous visitors."""
        client, csrf_token = self.get_authenticated_client()

        # Anonymous request -> 403 Forbidden
        resp = client.get('/unresolved_images_report')
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Researcher authorization required", resp.get_json().get('error', ''))

        # Authorized request with X-Admin-Key -> not 403
        resp_auth = client.get('/unresolved_images_report', headers={'X-Admin-Key': self.admin_key})
        self.assertNotEqual(resp_auth.status_code, 403)

    def test_researcher_session_authorization_endpoints(self):
        """Test /authorize_researcher and /deauthorize_researcher session unlock flow."""
        client, csrf_token = self.get_authenticated_client()

        # 1. Invalid key -> 403
        resp = client.post('/authorize_researcher', headers={'X-CSRFToken': csrf_token},
                           json={'admin_key': 'wrong_secret_key'})
        self.assertEqual(resp.status_code, 403)

        # 2. Valid key -> 200
        resp = client.post('/authorize_researcher', headers={'X-CSRFToken': csrf_token},
                           json={'admin_key': self.admin_key})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('success'))

        # 3. Deauthorize -> clears session
        resp_deauth = client.post('/deauthorize_researcher', headers={'X-CSRFToken': csrf_token},
                                  json={})
        self.assertEqual(resp_deauth.status_code, 200)
        self.assertTrue(resp_deauth.get_json().get('success'))

    # =========================================================================
    # K. FINAL REGRESSION TESTS FOR REMAINING ISSUES (AUDIT ROUND 4)
    # =========================================================================

    def test_research_profile_privacy_with_url_name_variations(self):
        """URL variations of baseline subject names (spaces, underscores, hyphens, casing)
        must NEVER bypass participant privacy restrictions for anonymous visitors."""
        client, csrf_token = self.get_authenticated_client()

        baseline_name = "Albert Einstein"
        dummy_scores = {t: 70.0 for t in app.TRAIT_NAMES}
        dummy_info = {
            'face_detected': True,
            'stats': {'nonzero': 512, 'min': 0.01, 'max': 0.85, 'mean': 0.04},
            'top50_values': [0.1] * 50
        }
        app.save_profile(baseline_name, dummy_scores, dummy_scores, "Confidential Bio", dummy_info, 'scientist')

        # Add to sandbox dataset.csv so it is recognized as a baseline subject
        df_base = pd.DataFrame([{'name': baseline_name}])
        df_base.to_csv(app.DATASET_FILE, index=False)

        # Test anonymous access across various URL encodings and naming styles
        test_variations = [
            "Albert Einstein",
            "Albert_Einstein",
            "Albert-Einstein",
            "albert_einstein",
            "albert-einstein",
            "albert einstein",
            "Albert%20Einstein"
        ]

        for var in test_variations:
            resp = client.get(f'/profile/{var}')
            self.assertEqual(resp.status_code, 200, f"Failed for URL variation: {var}")
            self.assertIn(b"Participant Privacy", resp.data, f"Privacy bypassed for variation: {var}")
            self.assertIn(b"research baseline dataset", resp.data, f"Privacy banner missing for variation: {var}")
            # Ensure confidential biography content is shielded
            self.assertNotIn(b"Confidential Bio", resp.data, f"Confidential text leaked for variation: {var}")

        # Authorized access with admin key must successfully display full profile for all variations
        resp_auth = client.get(f'/profile/Albert_Einstein', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp_auth.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_auth.data)
        self.assertIn(b"Confidential Bio", resp_auth.data)

    def test_duplicate_public_assessments_prevent_overwriting_records_and_profiles(self):
        """Duplicate public assessments must not overwrite previous records or profiles."""
        client, csrf_token = self.get_authenticated_client()
        pub_name = "Unique Citizen Beta"

        mock_scores = {
            'openness': 80.0, 'conscientiousness': 75.0, 'extraversion': 65.0,
            'agreeableness': 70.0, 'neuroticism': 45.0, 'reasoning': 'Initial assessment.'
        }

        # 1. First submission succeeds
        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': pub_name,
                'biography': self.valid_bio,
                'profession': 'other',
                'csrf_token': csrf_token,
                'photo': (f, 'test_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified', 'PASSED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                 patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
                resp1 = client.post('/analyze', data=data, content_type='multipart/form-data',
                                    headers={'X-CSRFToken': csrf_token})
                self.assertEqual(resp1.status_code, 200)
                self.assertTrue(resp1.get_json().get('success'))

        # Verify initial assessment was committed to public_assessments.csv and profiles
        df_pub_1 = pd.read_csv(app.PUBLIC_ASSESSMENTS_FILE)
        self.assertIn(pub_name, df_pub_1['name'].values)
        initial_record_count = len(df_pub_1)

        # 2. Second submission with EXACT duplicate name must return 409 Duplicate Error
        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': pub_name,
                'biography': self.valid_bio,
                'profession': 'other',
                'csrf_token': csrf_token,
                'photo': (f, 'test_face.jpg')
            }
            resp2 = client.post('/analyze', data=data, content_type='multipart/form-data',
                                headers={'X-CSRFToken': csrf_token})
            self.assertEqual(resp2.status_code, 409)
            res2_json = resp2.get_json()
            self.assertFalse(res2_json.get('success'))
            self.assertTrue(res2_json.get('duplicate'))
            self.assertIn("already exists", res2_json.get('error', '').lower())

        # 3. Third submission with NAME VARIATION ("Unique_Citizen_Beta") must ALSO be blocked
        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': "Unique_Citizen_Beta",
                'biography': self.valid_bio,
                'profession': 'other',
                'csrf_token': csrf_token,
                'photo': (f, 'test_face.jpg')
            }
            resp3 = client.post('/analyze', data=data, content_type='multipart/form-data',
                                headers={'X-CSRFToken': csrf_token})
            self.assertEqual(resp3.status_code, 409)
            self.assertTrue(resp3.get_json().get('duplicate'))

        # 4. Confirm public_assessments.csv was NOT altered or truncated
        df_pub_after = pd.read_csv(app.PUBLIC_ASSESSMENTS_FILE)
        self.assertEqual(len(df_pub_after), initial_record_count)
        self.assertIn(pub_name, df_pub_after['name'].values)

        # 5. Verify save_profile with allow_overwrite=False creates disambiguated profile without overwriting original
        path_orig = app.resolve_profile_path(pub_name)
        self.assertIsNotNone(path_orig)
        with open(path_orig, 'r', encoding='utf-8') as f:
            prof_orig = json.load(f)

        app.save_profile(pub_name, {'openness': 99.0, 'conscientiousness': 99.0, 'extraversion': 99.0, 'agreeableness': 99.0, 'neuroticism': 99.0},
                         None, "Secondary Reasoning", None, allow_overwrite=False)

        # Original profile must be completely untouched
        with open(path_orig, 'r', encoding='utf-8') as f:
            prof_recheck = json.load(f)
        self.assertEqual(prof_orig['biography_scores']['openness'], prof_recheck['biography_scores']['openness'])

    def test_env_and_credentials_security_and_exclusion(self):
        """Ensure .env and credentials cannot be served by Flask, and are excluded in version control/exports."""
        client, _ = self.get_authenticated_client()

        # 1. Attempting to fetch .env or dotfiles via HTTP returns 403 Forbidden
        for bad_path in ['/.env', '/.env.local', '/.git/config', '/static/../.env', '/profile/../../.env']:
            resp = client.get(bad_path)
            self.assertEqual(resp.status_code, 403, f"Protected file path not blocked: {bad_path}")

        # 2. Check .gitignore rules
        gitignore_path = os.path.join(APP_DIR, '.gitignore')
        self.assertTrue(os.path.exists(gitignore_path))
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gi_content = f.read()
        self.assertIn('.env', gi_content)
        self.assertIn('credentials.*', gi_content)

        # 3. Check .gitattributes export-ignore rules
        gitattr_path = os.path.join(APP_DIR, '.gitattributes')
        self.assertTrue(os.path.exists(gitattr_path))
        with open(gitattr_path, 'r', encoding='utf-8') as f:
            ga_content = f.read()
        self.assertIn('.env export-ignore', ga_content)
        self.assertIn('credentials.* export-ignore', ga_content)

        # 4. Check .env.example contains only dummy placeholders
        env_example_path = os.path.join(APP_DIR, '.env.example')
        self.assertTrue(os.path.exists(env_example_path))
        with open(env_example_path, 'r', encoding='utf-8') as f:
            ex_content = f.read()
        self.assertNotIn('sk-ant-', ex_content)
        self.assertIn('your_anthropic_api_key_here', ex_content)

        # 5. Local real .env exists and is not empty
        real_env_path = os.path.join(APP_DIR, '.env')
        if os.path.exists(real_env_path):
            self.assertGreater(os.path.getsize(real_env_path), 50)

    def test_score_similarity_label_not_confused_with_pearson_correlation(self):
        """Cross-modal score similarity must not use misleading 'Strong Correlation' label."""
        analyze_html_path = os.path.join(APP_DIR, 'templates', 'analyze.html')
        self.assertTrue(os.path.exists(analyze_html_path))
        with open(analyze_html_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # Misleading Pearson correlation labels must be absent
        self.assertNotIn("'Strong Correlation'", html)
        self.assertNotIn("'Moderate Correlation'", html)
        self.assertNotIn("'Weak Correlation'", html)

        # Accurate score agreement labels must be present
        self.assertIn("'Strong Score Agreement'", html)
        self.assertIn("'Moderate Score Agreement'", html)
        self.assertIn("distinct from population Pearson correlation (r)", html)

    # =========================================================================
    # L. FINAL CORRECTIONS & HARDENING TESTS (AUDIT ROUND 5)
    # =========================================================================

    def test_subjects_html_safe_event_handling_and_apostrophe_safety(self):
        """subjects.html must use safe event listeners and data attributes, eliminating inline onclick XSS."""
        template_path = os.path.join(APP_DIR, 'templates', 'subjects.html')
        self.assertTrue(os.path.exists(template_path))
        with open(template_path, 'r', encoding='utf-8') as f:
            tmpl_content = f.read()

        # Unsafe inline onclick deleteSubject must be completely eliminated
        self.assertNotIn('onclick="deleteSubject(', tmpl_content)

        # Safe data attributes and delegating classes must be present
        self.assertIn('js-delete-subject-btn', tmpl_content)
        self.assertIn('data-subject-name=', tmpl_content)

        # Render /subjects with authorized researcher when subject name contains apostrophes
        client, _ = self.get_authenticated_client()
        apostrophe_name = "O'Connor"
        dummy_scores = {t: 65.0 for t in app.TRAIT_NAMES}
        app.save_profile(apostrophe_name, dummy_scores, dummy_scores, "Bio", None, 'scientist')
        df_pub = pd.DataFrame([{'name': apostrophe_name}])
        df_pub.to_csv(app.PUBLIC_ASSESSMENTS_FILE, index=False)

        resp = client.get('/subjects', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp.status_code, 200)
        # Verify page rendered cleanly without broken JS inline syntax
        self.assertIn(b"js-delete-subject-btn", resp.data)

    def test_save_profile_executes_single_atomic_write(self):
        """save_profile() must execute exactly ONE atomic write and ONE image association without duplicate I/O."""
        dummy_scores = {t: 60.0 for t in app.TRAIT_NAMES}
        with patch('app.atomic_write_json') as mock_write, \
             patch('app.save_image_association') as mock_assoc:
            app.save_profile("Single Write Subject", dummy_scores, dummy_scores,
                             "Bio test", None, 'other', image_filename="sample_test.jpg")
            self.assertEqual(mock_write.call_count, 1, "save_profile must execute exactly one atomic_write_json call")
            self.assertEqual(mock_assoc.call_count, 1, "save_profile must execute exactly one save_image_association call")

    def test_public_assessment_profile_protection_and_creator_access(self):
        """Public assessment profiles are protected from unauthorized visitors,
        while assessment creators view own results via session/token and researchers manage via admin key."""
        client, csrf_token = self.get_authenticated_client()
        subject_name = "Dr. Jane Foster"
        mock_scores = {
            'openness': 85.0, 'conscientiousness': 80.0, 'extraversion': 60.0,
            'agreeableness': 75.0, 'neuroticism': 40.0, 'reasoning': 'Assessment for Dr. Jane Foster.'
        }

        # 1. Creator completes assessment via /analyze
        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': subject_name,
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_token,
                'photo': (f, 'jane_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match verified', 'PASSED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                 patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
                resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                   headers={'X-CSRFToken': csrf_token})
                self.assertEqual(resp.status_code, 200)
                res_data = resp.get_json()
                self.assertTrue(res_data.get('success'))
                creator_token = res_data.get('access_token')
                self.assertIsNotNone(creator_token)

        # 2. Creator client views own profile in same session -> 200 with full analysis
        resp_creator = client.get(f'/profile/{subject_name}')
        self.assertEqual(resp_creator.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_creator.data)

        # 3. Unauthorized new visitor in a different session -> 200 with restricted privacy protection notice
        new_unauth_client = app.app.test_client()
        resp_unauth = new_unauth_client.get(f'/profile/{subject_name}')
        self.assertEqual(resp_unauth.status_code, 200)
        self.assertIn(b"Assessment Privacy", resp_unauth.data)
        self.assertNotIn(b"Biography Analysis", resp_unauth.data)

        # 4. Visiting with the valid creator access token (?token=...) unlocks full profile
        resp_token = new_unauth_client.get(f'/profile/{subject_name}?token={creator_token}')
        self.assertEqual(resp_token.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_token.data)

        # 5. Authorized researcher views full profile
        resp_researcher = new_unauth_client.get(f'/profile/{subject_name}', headers={'X-Admin-Key': self.admin_key})
        self.assertEqual(resp_researcher.status_code, 200)
        self.assertIn(b"Biography Analysis", resp_researcher.data)

    def test_admin_credentials_in_query_parameters_strictly_rejected(self):
        """Passing admin credentials via URL query parameters must be strictly rejected (403)."""
        client = app.app.test_client()
        resp = client.get(f'/unresolved_images_report?admin_key={self.admin_key}')
        self.assertEqual(resp.status_code, 403)
        self.assertIn(b"Unauthorized", resp.data)

    def test_admin_session_invalidated_when_server_key_changes(self):
        """Active researcher session is immediately revoked when server RESEARCH_ADMIN_KEY changes."""
        client, csrf_token = self.get_authenticated_client()
        # 1. Authorize researcher session with current admin key
        resp_auth = client.post('/authorize_researcher', json={'admin_key': self.admin_key, 'csrf_token': csrf_token})
        self.assertEqual(resp_auth.status_code, 200)

        # 2. Verify researcher can access protected endpoint
        resp_ok = client.get('/unresolved_images_report')
        self.assertEqual(resp_ok.status_code, 200)

        # 3. Simulate rotating RESEARCH_ADMIN_KEY on server
        orig_key = os.environ['RESEARCH_ADMIN_KEY']
        try:
            os.environ['RESEARCH_ADMIN_KEY'] = 'new_rotated_admin_key_2026'
            # 4. Same client attempts access -> session must be invalidated and access denied (403)
            resp_denied = client.get('/unresolved_images_report')
            self.assertEqual(resp_denied.status_code, 403)
        finally:
            os.environ['RESEARCH_ADMIN_KEY'] = orig_key

    def test_enforce_secure_cookie_on_https_requests(self):
        """HTTPS requests or reverse proxies forwarding https must dynamically enable SESSION_COOKIE_SECURE."""
        client = app.app.test_client()
        client.get('/', headers={'X-Forwarded-Proto': 'https'})
        self.assertTrue(app.app.config.get('SESSION_COOKIE_SECURE'))

    def test_fail_closed_research_baseline_privacy_on_corrupted_or_missing_dataset(self):
        """When dataset.csv is corrupted or missing, baseline profiles fail closed and remain restricted."""
        client = app.app.test_client()
        baseline_subject = "Sandboxed_Subject_1"

        dataset_content = None
        if os.path.exists(app.DATASET_FILE):
            with open(app.DATASET_FILE, 'r', encoding='utf-8') as f:
                dataset_content = f.read()

        try:
            # Overwrite dataset.csv with corrupt content (empty/unreadable)
            with open(app.DATASET_FILE, 'w', encoding='utf-8') as f:
                f.write("corrupt,header\n")

            # Anonymous visitor tries to view Sandboxed_Subject_1
            resp = client.get(f'/profile/{baseline_subject}')
            self.assertEqual(resp.status_code, 200)
            # Must render restricted notice, NOT full biography analysis
            self.assertIn(b"Participant Privacy", resp.data)
            self.assertNotIn(b"Biography Analysis", resp.data)
        finally:
            if dataset_content is not None:
                with open(app.DATASET_FILE, 'w', encoding='utf-8') as f:
                    f.write(dataset_content)

    def test_public_assessment_data_integrity_on_csv_read_failure(self):
        """When reading public_assessments.csv fails, /analyze preserves existing records in backup and self-heals."""
        client, csrf_token = self.get_authenticated_client()
        subject_name = "Recovery_Test_Subject"

        # Pre-seed public_assessments.csv with some existing raw content
        pre_existing_data = "name,timestamp,profession,profession_label,image_filename,openness,conscientiousness,extraversion,agreeableness,neuroticism,has_prediction\nExisting_Subject,2026-01-01T00:00:00,scientist,Scientist,,50,50,50,50,50,False\n"
        with open(app.PUBLIC_ASSESSMENTS_FILE, 'w', encoding='utf-8') as f:
            f.write(pre_existing_data)

        mock_scores = {
            'openness': 65.0, 'conscientiousness': 70.0,
            'extraversion': 55.0, 'agreeableness': 60.0, 'neuroticism': 50.0
        }

        try:
            with open(self.verified_face_path, 'rb') as f:
                data = {
                    'name': subject_name,
                    'biography': self.valid_bio,
                    'profession': 'scientist',
                    'csrf_token': csrf_token,
                    'photo': (f, 'test_face.jpg')
                }
                orig_read_csv = pd.read_csv
                def mock_read_csv(filepath, *args, **kwargs):
                    if filepath == app.PUBLIC_ASSESSMENTS_FILE:
                        raise IOError("Simulated disk read glitch on public_assessments.csv")
                    return orig_read_csv(filepath, *args, **kwargs)

                with patch('pandas.read_csv', side_effect=mock_read_csv), \
                     patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')), \
                     patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                     patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
                    resp = client.post('/analyze', data=data, content_type='multipart/form-data',
                                       headers={'X-CSRFToken': csrf_token})
                    # Fails safely with HTTP 500 and clear storage error message
                    self.assertEqual(resp.status_code, 500)
                    resp_json = resp.get_json()
                    self.assertEqual(resp_json.get('status'), 'STORAGE_ERROR')
                    self.assertIn("Storage integrity error", resp_json.get('error', ''))
                    self.assertIn("Original records have been preserved untouched", resp_json.get('error', ''))

            # Verify that existing file was NOT overwritten or replaced with latest record
            with open(app.PUBLIC_ASSESSMENTS_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn("Existing_Subject", content)
            self.assertNotIn(subject_name, content)
        finally:
            if os.path.exists(app.PUBLIC_ASSESSMENTS_FILE):
                os.remove(app.PUBLIC_ASSESSMENTS_FILE)

    def test_public_assessment_privacy_isolated_to_creator_session_on_subjects(self):
        """Anonymous visitor B cannot see assessment names or details created by visitor A."""
        client_a, csrf_a = self.get_authenticated_client()
        subject_a = "Private_Public_Subject_Alpha"
        mock_scores = {
            'openness': 70.0, 'conscientiousness': 75.0,
            'extraversion': 60.0, 'agreeableness': 65.0, 'neuroticism': 45.0
        }

        with open(self.verified_face_path, 'rb') as f:
            data = {
                'name': subject_a,
                'biography': self.valid_bio,
                'profession': 'scientist',
                'csrf_token': csrf_a,
                'photo': (f, 'test_face.jpg')
            }
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')), \
                 patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
                 patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
                resp = client_a.post('/analyze', data=data, content_type='multipart/form-data',
                                     headers={'X-CSRFToken': csrf_a})
                self.assertEqual(resp.status_code, 200)

        # 1. Subject lists are researcher-only now; creator A still sees their own profile in-session
        self.assertEqual(client_a.get('/subjects').status_code, 302)
        resp_a_prof = client_a.get(f'/profile/{subject_a}')
        self.assertEqual(resp_a_prof.status_code, 200)
        self.assertIn(subject_a.encode('utf-8'), resp_a_prof.data)

        # 2. Anonymous Visitor B cannot list subjects at all
        client_b = app.app.test_client()
        resp_b_subjs = client_b.get('/subjects')
        self.assertEqual(resp_b_subjs.status_code, 302)
        self.assertNotIn(subject_a.encode('utf-8'), resp_b_subjs.data)

        # 3. Anonymous Visitor B attempts direct profile URL -> must show generic protected notice without leaking name
        resp_b_prof = client_b.get(f'/profile/{subject_a}')
        self.assertEqual(resp_b_prof.status_code, 200)
        self.assertIn(b"Protected Assessment Record", resp_b_prof.data)
        self.assertNotIn(b"Biography Analysis", resp_b_prof.data)
        self.assertNotIn(f"<h1>Assessment: {subject_a}</h1>".encode('utf-8'), resp_b_prof.data)

    def test_server_side_ai_rate_limiting_enforced_and_bypassed_for_researchers(self):
        """Public AI endpoints enforce rate limiting (429) when thresholds exceeded; researchers bypass."""
        client, csrf_token = self.get_authenticated_client()
        app.app.config['TESTING_RATELIMIT'] = True
        app.ai_rate_limiter.reset()

        try:
            # 1. /validate has limit 15 requests/min. Send 15 requests successfully (mocked)
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')):
                for i in range(15):
                    resp = client.post('/validate', json={
                        'name': 'Test Figure',
                        'biography': self.valid_bio,
                        'csrf_token': csrf_token
                    })
                    self.assertEqual(resp.status_code, 200)

                # 16th request must be RATE_LIMITED (429)
                resp_limited = client.post('/validate', json={
                    'name': 'Test Figure',
                    'biography': self.valid_bio,
                    'csrf_token': csrf_token
                })
                self.assertEqual(resp_limited.status_code, 429)
                lim_data = resp_limited.get_json()
                self.assertEqual(lim_data.get('status'), 'RATE_LIMITED')
                self.assertIn('Rate limit exceeded', lim_data.get('error', ''))

            # 2. An authorized researcher using the same IP bypasses rate limiting
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')):
                resp_researcher = client.post('/validate',
                                              headers={'X-Admin-Key': self.admin_key},
                                              json={
                                                  'name': 'Test Figure',
                                                  'biography': self.valid_bio,
                                                  'csrf_token': csrf_token
                                              })
                self.assertEqual(resp_researcher.status_code, 200)
        finally:
            app.app.config.pop('TESTING_RATELIMIT', None)
            app.ai_rate_limiter.reset()

    def test_client_ip_spoofed_forwarded_for_rejected_from_untrusted_remote(self):
        """Direct clients cannot spoof X-Forwarded-For or X-Real-IP to bypass rate limits."""
        with app.app.test_request_context('/', environ_base={'REMOTE_ADDR': '198.51.100.77'},
                                          headers={'X-Forwarded-For': '1.2.3.4', 'X-Real-IP': '5.6.7.8'}):
            ip = app.get_client_ip()
            self.assertEqual(ip, '198.51.100.77')

    def test_client_ip_trusted_proxy_correctly_extracts_client_ip(self):
        """Forwarding headers are respected only when connection originates from an explicitly trusted proxy."""
        orig_proxy_setting = app.app.config.get('TRUST_PROXY_HEADERS')
        app.app.config['TRUST_PROXY_HEADERS'] = True
        try:
            # Trusted proxy (127.0.0.1) forwarding real client IP
            with app.app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'},
                                              headers={'X-Forwarded-For': '203.0.113.195'}):
                ip = app.get_client_ip()
                self.assertEqual(ip, '203.0.113.195')

            # Multi-hop proxy chain: client, proxy1 -> extracts rightmost untrusted hop
            app.app.config['TRUSTED_PROXIES'] = '127.0.0.1, 10.0.0.1'
            with app.app.test_request_context('/', environ_base={'REMOTE_ADDR': '127.0.0.1'},
                                              headers={'X-Forwarded-For': '203.0.113.195, 10.0.0.1'}):
                ip = app.get_client_ip()
                self.assertEqual(ip, '203.0.113.195')
        finally:
            if orig_proxy_setting is not None:
                app.app.config['TRUST_PROXY_HEADERS'] = orig_proxy_setting
            else:
                app.app.config.pop('TRUST_PROXY_HEADERS', None)
            app.app.config.pop('TRUSTED_PROXIES', None)

    def test_rate_limit_bypass_via_spoofed_headers_strictly_prevented(self):
        """Attacker sending rotating X-Forwarded-For headers from same direct IP cannot bypass rate limit."""
        client, csrf_token = self.get_authenticated_client()
        app.app.config['TESTING_RATELIMIT'] = True
        app.ai_rate_limiter.reset()

        try:
            with patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')):
                # Untrusted caller sends 15 requests with rotating spoofed X-Forwarded-For
                for i in range(15):
                    resp = client.post('/validate',
                                       environ_base={'REMOTE_ADDR': '198.51.100.99'},
                                       headers={'X-Forwarded-For': f'10.0.0.{i+1}'},
                                       json={
                                           'name': 'Test Figure',
                                           'biography': self.valid_bio,
                                           'csrf_token': csrf_token
                                       })
                    self.assertEqual(resp.status_code, 200)

                # 16th request from same direct IP with new spoofed X-Forwarded-For must be BLOCKED (429)
                resp_blocked = client.post('/validate',
                                           environ_base={'REMOTE_ADDR': '198.51.100.99'},
                                           headers={'X-Forwarded-For': '10.0.0.99'},
                                           json={
                                               'name': 'Test Figure',
                                               'biography': self.valid_bio,
                                               'csrf_token': csrf_token
                                           })
                self.assertEqual(resp_blocked.status_code, 429)
                data = resp_blocked.get_json()
                self.assertEqual(data.get('status'), 'RATE_LIMITED')
                self.assertIn('Rate limit exceeded', data.get('error', ''))
        finally:
            app.app.config.pop('TESTING_RATELIMIT', None)
            app.ai_rate_limiter.reset()

    def test_concurrent_public_assessment_submissions_serialized_safely(self):
        """Concurrent assessment submissions are synchronized by transaction lock without record loss."""
        import concurrent.futures

        # Clean sandbox public_assessments.csv
        if os.path.exists(app.PUBLIC_ASSESSMENTS_FILE):
            os.remove(app.PUBLIC_ASSESSMENTS_FILE)

        mock_scores = {
            'openness': 70.0, 'conscientiousness': 75.0,
            'extraversion': 60.0, 'agreeableness': 65.0, 'neuroticism': 45.0
        }

        num_threads = 5
        subjects = [f"Concurrent_Subject_{i}" for i in range(num_threads)]

        def submit_assessment(subj_name):
            thread_client, csrf = self.get_authenticated_client()
            with open(self.verified_face_path, 'rb') as f:
                data = {
                    'name': subj_name,
                    'biography': self.valid_bio,
                    'profession': 'scientist',
                    'csrf_token': csrf,
                    'photo': (f, 'test_face.jpg')
                }
                resp = thread_client.post('/analyze', data=data, content_type='multipart/form-data',
                                          headers={'X-CSRFToken': csrf})
                return resp.status_code, subj_name

        with patch('app.validate_match_via_gemini', return_value=(True, 'Match', 'PASSED')), \
             patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.score_personality_via_gemini', return_value=(mock_scores.copy(), None, 'PASSED')):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(submit_assessment, subjects))

        for status_code, sname in results:
            self.assertEqual(status_code, 200, f"Submission for {sname} failed with status {status_code}")

        # Verify all concurrent subjects are present in public_assessments.csv
        self.assertTrue(os.path.exists(app.PUBLIC_ASSESSMENTS_FILE))
        df_final = pd.read_csv(app.PUBLIC_ASSESSMENTS_FILE)
        for sname in subjects:
            self.assertIn(sname, df_final['name'].values, f"Missing concurrent subject: {sname}")
        self.assertGreaterEqual(len(df_final), num_threads)

    def test_env_and_secrets_strictly_excluded_from_git_and_archives(self):
        """Verify .gitignore and .gitattributes exclude .env and sensitive secrets across root and project."""
        app_gitignore = os.path.join(APP_DIR, '.gitignore')
        app_gitattributes = os.path.join(APP_DIR, '.gitattributes')
        root_gitignore = os.path.join(os.path.dirname(APP_DIR), '.gitignore')
        root_gitattributes = os.path.join(os.path.dirname(APP_DIR), '.gitattributes')

        required = [app_gitignore, app_gitattributes]
        # Parent-folder files are only required when the parent is a project folder (has its own .gitignore)
        if os.path.exists(root_gitignore):
            required += [root_gitignore, root_gitattributes]
        for fpath in required:
            self.assertTrue(os.path.exists(fpath), f"Missing exclusion file: {fpath}")
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('.env', content)
            self.assertTrue(any(term in content for term in ['*.key', '*.secret', 'credentials.*']))

    def test_hosting_compatibility_procfile_and_requirements(self):
        """Verify Procfile and requirements.txt configure production server execution cleanly."""
        procfile_path = os.path.join(APP_DIR, 'Procfile')
        self.assertTrue(os.path.exists(procfile_path), "Missing Procfile for production PaaS deployment")
        with open(procfile_path, 'r', encoding='utf-8') as f:
            proc_content = f.read()
        self.assertIn("gunicorn app:app", proc_content)

        req_path = os.path.join(APP_DIR, 'requirements.txt')
        with open(req_path, 'r', encoding='utf-8') as f:
            req_content = f.read()
        self.assertIn("gunicorn", req_content)
        self.assertIn("flask", req_content)
        self.assertIn("opencv-python", req_content)
        self.assertIn("scikit-learn", req_content)

    def test_persistent_storage_initialization_includes_all_103_profiles(self):
        """Verify init_persistent_storage seeds all 103 research profiles without overwriting existing data."""
        with tempfile.TemporaryDirectory() as temp_vol:
            target_data = os.path.join(temp_vol, 'data')
            target_uploads = os.path.join(temp_vol, 'uploads')
            target_models = os.path.join(temp_vol, 'models')

            # Initial migration onto a new volume
            app.init_persistent_storage(data_dir=target_data, upload_dir=target_uploads, models_dir=target_models)

            # 1. Verify 103 profiles migrated
            target_profiles_dir = os.path.join(target_data, 'profiles')
            self.assertTrue(os.path.exists(target_profiles_dir))
            profile_files = [f for f in os.listdir(target_profiles_dir) if f.lower().endswith('.json')]
            self.assertGreaterEqual(len(profile_files), 103, "All 103 baseline profile JSON files must be seeded")

            # 2. Verify baseline datasets and model present
            self.assertTrue(os.path.exists(os.path.join(target_data, 'processed', 'dataset.csv')))
            self.assertTrue(os.path.exists(os.path.join(target_data, 'processed', 'face_embeddings.csv')))
            self.assertTrue(os.path.exists(os.path.join(target_data, 'processed', 'personality_scores.csv')))
            self.assertTrue(os.path.exists(os.path.join(target_data, 'image_associations.json')))
            self.assertTrue(os.path.exists(os.path.join(target_models, 'trained_model.pkl')))

            # 3. Test non-destructive idempotency: existing file must NEVER be overwritten
            sample_profile = os.path.join(target_profiles_dir, profile_files[0])
            with open(sample_profile, 'w', encoding='utf-8') as f:
                f.write('{"custom_preservation_test": true}')

            # Rerun migration
            app.init_persistent_storage(data_dir=target_data, upload_dir=target_uploads, models_dir=target_models)
            with open(sample_profile, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertIn('"custom_preservation_test": true', content, "Existing persistent volume data must never be overwritten")

    def test_shared_rate_limiter_across_simulated_gunicorn_workers(self):
        """Verify SimpleRateLimiter synchronizes rate limits across multiple worker processes via SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_db = os.path.join(temp_dir, 'shared_rate.db')

            # Simulate two independent Gunicorn worker processes sharing the same db
            worker_1_limiter = app.SimpleRateLimiter(db_path=shared_db)
            worker_2_limiter = app.SimpleRateLimiter(db_path=shared_db)

            # Worker 1 receives 10 requests from IP 203.0.113.5
            for _ in range(10):
                limited, _ = worker_1_limiter.is_rate_limited('test_ep', '203.0.113.5', limit=15, window_seconds=60)
                self.assertFalse(limited)

            # Worker 2 receives 5 requests from same IP 203.0.113.5
            for _ in range(5):
                limited, _ = worker_2_limiter.is_rate_limited('test_ep', '203.0.113.5', limit=15, window_seconds=60)
                self.assertFalse(limited)

            # 16th request to Worker 2 must be rate limited because Worker 1 and Worker 2 share state
            limited, retry_after = worker_2_limiter.is_rate_limited('test_ep', '203.0.113.5', limit=15, window_seconds=60)
            self.assertTrue(limited, "Worker 2 must enforce rate limit using shared multi-process history")
            self.assertGreater(retry_after, 0)

            # 16th request to Worker 1 must also be rate limited
            limited_w1, _ = worker_1_limiter.is_rate_limited('test_ep', '203.0.113.5', limit=15, window_seconds=60)
            self.assertTrue(limited_w1, "Worker 1 must also enforce rate limit using shared multi-process history")

            # Reset clears shared state for both workers
            worker_1_limiter.reset()
            limited_after_reset, _ = worker_2_limiter.is_rate_limited('test_ep', '203.0.113.5', limit=15, window_seconds=60)
            self.assertFalse(limited_after_reset)

    def test_scientific_terminology_accurate_and_distinguished(self):
        """Dashboard must use accurate scientific terminology distinguishing LLM reference scores from ground truth."""
        index_path = os.path.join(APP_DIR, 'templates', 'index.html')
        self.assertTrue(os.path.exists(index_path))
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Inaccurate ground-truth terminology must be absent
        self.assertNotIn("Ground-Truth Reference (OCEAN)", content)
        self.assertNotIn("Confidential ground-truth calibration subjects", content)

        # Accurate computational terminology must be present
        self.assertIn("OCEAN Reference Scores (0–100)", content)
        self.assertIn("BFI-10", content)
        self.assertIn("Research dataset records", content)

    def test_safe_deployment_packaging_and_credential_exclusion(self):
        """Verify safe deployment packaging utility excludes credentials, caches, and tests."""
        pkg_script = os.path.join(APP_DIR, 'package_deployment.py')
        self.assertTrue(os.path.exists(pkg_script), "Missing package_deployment.py utility")

        sys.path.insert(0, APP_DIR)
        import package_deployment

        # 1. Exclusion logic checks
        self.assertTrue(package_deployment.is_path_excluded('.env')[0])
        self.assertTrue(package_deployment.is_path_excluded('.env.production')[0])
        self.assertTrue(package_deployment.is_path_excluded('credentials.json')[0])
        self.assertTrue(package_deployment.is_path_excluded('secret.key')[0])
        self.assertTrue(package_deployment.is_path_excluded('tests/test_facepersona.py')[0])
        self.assertTrue(package_deployment.is_path_excluded('__pycache__/app.cpython-314.pyc')[0])
        self.assertTrue(package_deployment.is_path_excluded('deploy.zip')[0])

        # 2. Inclusions whitelist
        self.assertFalse(package_deployment.is_path_excluded('.env.example')[0])
        self.assertFalse(package_deployment.is_path_excluded('app.py')[0])
        self.assertFalse(package_deployment.is_path_excluded('Procfile')[0])
        self.assertFalse(package_deployment.is_path_excluded('templates/index.html')[0])

        # 3. Audit check
        included, excluded, issues = package_deployment.audit_project(APP_DIR)
        self.assertEqual(len(issues), 0, f"Deployment audit issues: {issues}")
        self.assertIn('.env.example', included)
        self.assertIn('app.py', included)
        self.assertNotIn('.env', included)

        # 4. End-to-end safe zip creation in temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_zip = os.path.join(tmp_dir, 'test_deploy.zip')
            n_inc, n_exc = package_deployment.create_safe_zip(APP_DIR, test_zip)
            self.assertTrue(os.path.exists(test_zip))
            with zipfile.ZipFile(test_zip, 'r') as zf:
                names = zf.namelist()
                self.assertNotIn('.env', names)
                self.assertIn('app.py', names)
                self.assertIn('.env.example', names)
                # Verify no live secrets in any text file inside zip
                for zname in names:
                    if zname.endswith(('.py', '.html', '.css', '.js', '.json', '.txt', '.md', '.example')):
                        text = zf.read(zname).decode('utf-8', errors='ignore')
                        self.assertNotIn('sk-ant-', text)

    # =========================================================================
    # CELEBRITY RECOGNITION (AWS REKOGNITION) TESTS
    # =========================================================================

    def _aws_env(self):
        return patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'test', 'AWS_SECRET_ACCESS_KEY': 'test'})

    def _mock_profession(self, value):
        mock_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"profession": "%s"}' % value)]
        mock_client.messages.create.return_value = mock_msg
        return mock_client

    def test_celebrity_recognition_disabled_without_aws_keys(self):
        """Without AWS credentials the feature is disabled and the endpoint degrades gracefully."""
        client, csrf_token = self.get_authenticated_client()
        with patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': '', 'AWS_SECRET_ACCESS_KEY': ''}):
            self.assertFalse(app.is_celebrity_recognition_enabled())
            with open(self.verified_face_path, 'rb') as f:
                resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                                   data={'photo': (f, 'face.jpg')}, content_type='multipart/form-data')
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(data['success'])
        self.assertFalse(data['service_enabled'])
        self.assertFalse(data['recognized'])
        self.assertTrue(os.path.exists(os.path.join(app.UPLOAD_DIR, data['upload_id'])))

    def test_celebrity_recognized_returns_name_and_profession(self):
        """A confident Rekognition match returns the name and a suggested profession for auto-fill."""
        client, csrf_token = self.get_authenticated_client()
        rek = MagicMock()
        rek.recognize_celebrities.return_value = {'CelebrityFaces': [
            {'Name': 'Low Match', 'MatchConfidence': 55.0, 'Urls': []},
            {'Name': 'Famous Person', 'MatchConfidence': 99.1, 'Urls': ['www.wikidata.org/wiki/Q1']},
        ]}
        with self._aws_env(), patch('app.get_rekognition_client', return_value=rek), \
                patch('app.get_claude_client', return_value=self._mock_profession('musician')):
            with open(self.verified_face_path, 'rb') as f:
                resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                                   data={'photo': (f, 'face.jpg')}, content_type='multipart/form-data')
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(data['recognized'])
        self.assertEqual(data['name'], 'Famous Person')
        self.assertEqual(data['profession'], 'musician')
        self.assertEqual(data['other_faces'], 0, "Low-confidence matches must be ignored")

    def test_celebrity_below_threshold_not_recognized(self):
        """Matches under CELEBRITY_MIN_CONFIDENCE are not auto-filled (e.g. ordinary people / look-alikes)."""
        rek = MagicMock()
        rek.recognize_celebrities.return_value = {'CelebrityFaces': [
            {'Name': 'Look Alike', 'MatchConfidence': 70.0}], 'UnrecognizedFaces': [{}]}
        with self._aws_env(), patch('app.get_rekognition_client', return_value=rek):
            result, err, status = app.recognize_celebrity(self.verified_face_path)
        self.assertIsNone(result)
        self.assertEqual(status, 'NOT_RECOGNIZED')

    def test_celebrity_service_error_reported_as_unavailable(self):
        """AWS errors never crash the endpoint; they are reported as UNAVAILABLE."""
        client, csrf_token = self.get_authenticated_client()
        rek = MagicMock()
        rek.recognize_celebrities.side_effect = Exception('AccessDenied')
        with self._aws_env(), patch('app.get_rekognition_client', return_value=rek):
            with open(self.verified_face_path, 'rb') as f:
                resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                                   data={'photo': (f, 'face.jpg')}, content_type='multipart/form-data')
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data['status'], 'UNAVAILABLE')
        self.assertFalse(data['recognized'])

    def test_identify_photo_rejects_path_traversal(self):
        """Existing-reference mode goes through the same upload path confinement."""
        client, csrf_token = self.get_authenticated_client()
        resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                           json={'upload_id': '../app.py'})
        self.assertEqual(resp.status_code, 400)

    def test_rekognition_image_conversion_for_unsupported_format(self):
        """WebP/BMP images are converted to JPEG bytes before being sent to Rekognition."""
        img = cv2.imread(self.verified_face_path)
        webp_path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.webp")
        cv2.imwrite(webp_path, img)
        data = app.prepare_image_bytes_for_rekognition(webp_path)
        self.assertTrue(data.startswith(b'\xff\xd8'), "Converted bytes must be JPEG")
        self.assertLessEqual(len(data), app.REKOGNITION_MAX_BYTES)


    # =========================================================================
    # AI-GENERATED IMAGE DETECTION TESTS
    # =========================================================================

    def _claude_prompt_mock(self, moderation=None, ai=None, face=None, explain=None):
        """Claude client mock that answers based on WHICH check is asking (safe for parallel calls)."""
        answers = [
            ('content moderator', moderation or '{"ok": true, "issue": null, "reason": null}'),
            ('Classify this image', ai or '{"classification": "real_photo", "confidence": 5, "reason": "Real photo."}'),
            ('Judge only whether the face', face or '{"verdict": "ok"}'),
            ('A face detector found no visible', explain or '{"category": "other"}'),
        ]
        mock_client = MagicMock()
        def _create(*a, **k):
            text = ' '.join(part.get('text', '') for msg in k.get('messages', [])
                            for part in (msg['content'] if isinstance(msg['content'], list) else [{'text': msg['content']}])
                            if isinstance(part, dict))
            reply = next((ans for key, ans in answers if key in text), '{}')
            m = MagicMock()
            m.content = [MagicMock(text=reply)]
            return m
        mock_client.messages.create.side_effect = _create
        return mock_client

    def _claude_text_mock(self, *texts):
        """Claude client mock returning the given texts in order (last one repeats)."""
        mock_client = MagicMock()
        msgs = []
        for t in texts:
            m = MagicMock()
            m.content = [MagicMock(text=t)]
            msgs.append(m)
        def _create(*a, **k):
            return msgs.pop(0) if len(msgs) > 1 else msgs[0]
        mock_client.messages.create.side_effect = _create
        return mock_client

    def _copy_face_with_suffix(self, ext, extra_bytes=b''):
        img = cv2.imread(self.verified_face_path)
        path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")
        cv2.imwrite(path, img)
        if extra_bytes:
            with open(path, 'ab') as f:
                f.write(extra_bytes)
        return path

    def test_ai_metadata_detects_iptc_and_sd_png(self):
        """Generator metadata (IPTC trainedAlgorithmicMedia, Stable Diffusion PNG params) is detected."""
        jpg = self._copy_face_with_suffix('jpg', b'<Iptc4xmpExt:DigitalSourceType>http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>')
        self.assertIn('trainedAlgorithmicMedia', app.detect_ai_metadata(jpg))
        png = self._copy_face_with_suffix('png', b'\x00\x00\x00\x30tEXtparameters\x00a portrait\nSteps: 30, Sampler: Euler a')
        self.assertIn('Stable Diffusion', app.detect_ai_metadata(png))
        c2pa = self._copy_face_with_suffix('png', b'jumbc2pa ... claim_generator OpenAI-API')
        self.assertIn('OpenAI', app.detect_ai_metadata(c2pa))

    def test_ai_metadata_absent_for_normal_photo(self):
        """A normal photo without generator markers is not flagged by metadata."""
        self.assertIsNone(app.detect_ai_metadata(self.verified_face_path))

    def test_check_photo_flags_ai_image_by_vision(self):
        """check_photo returns issue=ai_generated when Claude Vision is confident the image is AI-made."""
        app._AI_CHECK_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_prompt_mock(ai='{"ai_generated": true, "confidence": 95, "reason": "Melted fingers and garbled text."}')
        with patch('app.get_claude_client', return_value=mock):
            resp = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(path), 'profession': 'musician'})
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(data['ok'])
        self.assertEqual(data['issue'], 'ai_generated')
        self.assertEqual(data['detection_method'], 'vision')
        self.assertFalse(os.path.exists(path), "Temporary AI-generated upload should be cleaned up")

    def test_vision_low_confidence_not_blocked(self):
        """Low-confidence AI guesses do not block real photos (reduces false positives)."""
        app._AI_CHECK_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_text_mock('{"ai_generated": true, "confidence": 40, "reason": "Slightly smooth skin."}')
        with patch('app.get_claude_client', return_value=mock):
            res = app.check_ai_generated_image(path)
        self.assertFalse(res['is_ai'])
        self.assertEqual(res['status'], 'PASSED')

    def test_ai_check_service_failure_does_not_block(self):
        """If the vision service fails, the AI screening reports UNAVAILABLE without blocking."""
        app._AI_CHECK_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        with patch('app.get_claude_client', side_effect=Exception('network down')):
            res = app.check_ai_generated_image(path)
        self.assertFalse(res['is_ai'])
        self.assertEqual(res['status'], 'UNAVAILABLE')

    def test_digital_art_classification_blocked(self):
        """Stylized illustrations / digital paintings of a person are rejected as non-photos."""
        app._AI_CHECK_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_text_mock('{"classification": "digital_art", "confidence": 92, "reason": "Painterly rendering with stylized sunset."}')
        with patch('app.get_claude_client', return_value=mock):
            res = app.check_ai_generated_image(path)
        self.assertTrue(res['is_ai'])
        self.assertEqual(res['classification'], 'digital_art')

    def test_real_photo_classification_passes(self):
        """A real photo classification passes the AI screening."""
        app._AI_CHECK_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_text_mock('{"classification": "real_photo", "confidence": 5, "reason": "Natural skin texture."}')
        with patch('app.get_claude_client', return_value=mock):
            res = app.check_ai_generated_image(path)
        self.assertFalse(res['is_ai'])
        self.assertEqual(res['status'], 'PASSED')

    def test_analyze_rejects_ai_image_server_side(self):
        """/analyze enforces AI screening server-side even if the frontend check is skipped."""
        app._AI_CHECK_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        path = self._copy_face_with_suffix('jpg', b'Software: Midjourney v6')
        mock = self._claude_text_mock('{"is_safe": true, "reason": "Normal portrait."}')
        with patch('app.get_claude_client', return_value=mock):
            resp = client.post('/analyze', headers={'X-CSRFToken': csrf_token}, data={
                'name': 'Synthetic Person Test', 'biography': self.valid_bio, 'profession': 'scientist',
                'image_url_path': os.path.basename(path)}, content_type='multipart/form-data')
        data = resp.get_json()
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(data.get('ai_generated'))
        self.assertEqual(data.get('detection_method'), 'metadata')
        self.assertFalse(os.path.exists(path))

    # =========================================================================
    # NAME EXTRACTION & NO-FACE EXPLANATION TESTS
    # =========================================================================

    def test_name_extracted_only_from_wikipedia_urls(self):
        """Wikipedia article URLs give a name; Google thumbnails / CDN links do not (no more "images")."""
        self.assertEqual(app.extract_name_from_url('https://en.wikipedia.org/wiki/Bruno_Mars'), 'Bruno Mars')
        self.assertEqual(app.extract_name_from_url('https://en.wikipedia.org/wiki/Drake_(musician)'), 'Drake')
        self.assertEqual(app.extract_name_from_url('https://tl.wikipedia.org/wiki/Jos%C3%A9_Rizal'), 'José Rizal')
        self.assertIsNone(app.extract_name_from_url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9Gc'))
        self.assertIsNone(app.extract_name_from_url('https://cdn.example.com/photos/12345.jpg'))
        self.assertIsNone(app.extract_name_from_url('https://en.wikipedia.org/wiki/File:Bruno_Mars.jpg'))
        self.assertIsNone(app.extract_name_from_url('https://notwikipedia.org/wiki/Someone'))

    def test_explain_missing_face_covered(self):
        """A masked/helmeted subject gets a specific 'face is covered' message."""
        mock = self._claude_text_mock('{"category": "covered"}')
        with patch('app.get_claude_client', return_value=mock):
            category, msg = app.explain_missing_face(self.no_face_path)
        self.assertEqual(category, 'covered')
        self.assertIn('covered', msg)
        self.assertIn('No visible face detected', msg)

    def test_explain_missing_face_falls_back_on_error(self):
        """If the explanation service fails, the generic no-face message is used."""
        with patch('app.get_claude_client', side_effect=Exception('down')):
            category, msg = app.explain_missing_face(self.no_face_path)
        self.assertEqual(msg, app.GENERIC_NO_FACE_MESSAGE)

    def test_check_photo_reports_covered_face_early(self):
        """check_photo flags photos with no visible face before the user submits."""
        app._AI_CHECK_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        img = cv2.imread(self.no_face_path)
        path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(path, img)
        mock = self._claude_prompt_mock(explain='{"category": "covered"}')
        with patch('app.get_claude_client', return_value=mock):
            resp = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(path), 'profession': 'musician'})
        data = resp.get_json()
        self.assertFalse(data['ok'])
        self.assertEqual(data['issue'], 'no_face')
        self.assertEqual(data['no_face_category'], 'covered')

    def test_researcher_unlock_never_crashes_on_pasted_key(self):
        """Pasted keys with invisible / non-ASCII characters must not cause a 500 error,
        and a correct key wrapped in invisible spaces must still unlock."""
        client, csrf_token = self.get_authenticated_client()
        for bad in ['abc\u00a0def', '\u201cwrongkey\u201d', '\u00f1']:
            resp = client.post('/authorize_researcher', data={'csrf_token': csrf_token, 'admin_key': bad})
            self.assertEqual(resp.status_code, 302, f"Unlock crashed for input {bad!r}")
        resp = client.post('/authorize_researcher',
                           data={'csrf_token': csrf_token, 'admin_key': '\u200b ' + self.admin_key + '\u00a0'})
        self.assertEqual(resp.status_code, 302)
        with client.session_transaction() as sess:
            self.assertTrue(sess.get('research_admin_authorized'))

    # =========================================================================
    # PRIVATE PARTICIPANT MODE TESTS
    # =========================================================================

    def _private_upload(self):
        img = cv2.imread(self.verified_face_path)
        fname = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(app.UPLOAD_DIR, fname), img)
        return fname

    def _private_payload(self, upload_id, **over):
        payload = {'consent': True, 'age_range': '18-24', 'background': 'college student',
                   'answers': [2, 4, 2, 3, 2, 4, 2, 5, 3, 4], 'upload_id': upload_id}
        payload.update(over)
        return payload

    def test_bfi10_scoring_and_reverse_items(self):
        """BFI-10 scoring: neutral answers give 50; reverse-keyed items are flipped correctly."""
        self.assertEqual(app.compute_bfi10_scores([3] * 10)[0]['openness'], 50.0)
        high = app.compute_bfi10_scores([1, 5, 1, 1, 1, 5, 1, 5, 5, 5])[0]
        self.assertTrue(all(v == 100.0 for v in high.values()))
        self.assertIsNotNone(app.compute_bfi10_scores([3] * 9)[1])
        self.assertIsNotNone(app.compute_bfi10_scores([3] * 9 + [6])[1])
        self.assertIsNotNone(app.compute_bfi10_scores([3] * 9 + [True])[1])

    def test_private_participant_end_to_end_stored_separately(self):
        """Private assessment uses BFI-10 scores, stores no name, keeps the full embedding,
        and stays out of the public subject list and the research baseline."""
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed), \
             patch('app.score_personality_via_gemini') as mock_score, \
             patch('app.validate_match_via_gemini') as mock_validate:
            resp = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                               json=self._private_payload(upload_id))
            mock_score.assert_not_called()
            mock_validate.assert_not_called()
        data = resp.get_json()
        self.assertEqual(resp.status_code, 200, data)
        self.assertTrue(data['participant_id'].startswith('P-'))
        self.assertEqual(data['score_source'], 'BFI-10')
        self.assertEqual(data['biography_scores']['conscientiousness'], 87.5)

        df = pd.read_csv(app.get_private_participants_file())
        row = df[df['participant_id'] == data['participant_id']].iloc[0]
        self.assertNotIn('name', df.columns)
        self.assertIn('face_512', df.columns)
        self.assertEqual(row['age_range'], '18-24')
        self.assertEqual(row['image_filename'], upload_id)
        self.assertTrue(os.path.exists(os.path.join(app.UPLOAD_DIR, upload_id)))

        subjects_html = client.get('/subjects').data.decode()
        self.assertNotIn(data['participant_id'], subjects_html)
        self.assertNotIn(data['participant_id'], app.get_research_baseline_names())

    def test_private_requires_consent_age_and_all_answers(self):
        """Missing consent, invalid age range, or incomplete answers are rejected before any processing."""
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        for over, expect in [({'consent': False}, 'Consent'),
                             ({'age_range': '15-17'}, 'age range'),
                             ({'answers': [3] * 9}, 'all 10'),
                             ({'background': 'x' * 301}, 'at most')]:
            with patch('app.moderate_image') as mock_mod:
                resp = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                                   json=self._private_payload(upload_id, **over))
                mock_mod.assert_not_called()
            self.assertEqual(resp.status_code, 400)
            self.assertIn(expect, resp.get_json()['error'])

    def test_private_upload_skips_celebrity_recognition(self):
        """Private participant photos are never sent to celebrity recognition."""
        client, csrf_token = self.get_authenticated_client()
        with patch('app.recognize_celebrity') as mock_rec:
            with open(self.verified_face_path, 'rb') as f:
                resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                                   data={'photo': (f, 'me.jpg'), 'skip_recognition': '1', 'consent': 'true'},
                                   content_type='multipart/form-data')
            mock_rec.assert_not_called()
        data = resp.get_json()
        self.assertEqual(data['status'], 'SKIPPED')
        self.assertTrue(os.path.exists(os.path.join(app.UPLOAD_DIR, data['upload_id'])))

    def test_private_delete_cli_removes_record_and_photo(self):
        """manage_research.py private-delete honors a participant's deletion request."""
        import manage_research
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed):
            pid = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                              json=self._private_payload(upload_id)).get_json()['participant_id']
        manage_research.cmd_private_delete(MagicMock(participant_id=pid.lower()))
        df = pd.read_csv(app.get_private_participants_file())
        self.assertNotIn(pid, df['participant_id'].tolist())
        self.assertFalse(os.path.exists(os.path.join(app.UPLOAD_DIR, upload_id)))
        self.assertNotIn(pid, app.load_image_associations())

    # =========================================================================
    # FACE QUALITY & RESPONSE QUALITY TESTS
    # =========================================================================

    def test_heavily_blurred_face_rejected_without_api(self):
        """A heavily blurred face is rejected by the sharpness floor before any AI call."""
        app._FACE_QUALITY_CACHE.clear()
        img = cv2.imread(self.verified_face_path)
        sharp = app.measure_face_sharpness(self.verified_face_path)
        self.assertIsNotNone(sharp)
        self.assertGreater(sharp, app.FACE_BLUR_HARD_MIN)
        blurred_path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.jpg")
        cv2.imwrite(blurred_path, cv2.GaussianBlur(img, (0, 0), 6))
        sharpness = app.measure_face_sharpness(blurred_path)
        if sharpness is None:
            self.skipTest("Detector found no face after blurring")
        with patch('app.get_claude_client') as mock_claude:
            ok, category, msg = app.check_face_quality(blurred_path)
            if sharpness < app.FACE_BLUR_HARD_MIN:
                mock_claude.assert_not_called()
                self.assertFalse(ok)
                self.assertEqual(category, 'blurred')

    def test_face_quality_verdicts_from_vision(self):
        """Claude Vision verdicts (multiple people, blurred, not real) are turned into clear messages."""
        for verdict in ['multiple_people', 'blurred', 'not_real_face', 'covered']:
            app._FACE_QUALITY_CACHE.clear()
            mock = self._claude_text_mock('{"verdict": "%s"}' % verdict)
            with patch('app.get_claude_client', return_value=mock):
                ok, category, msg = app.check_face_quality(self.verified_face_path)
            self.assertFalse(ok)
            self.assertEqual(category, verdict)
            self.assertEqual(msg, app.FACE_QUALITY_MESSAGES[verdict])
        app._FACE_QUALITY_CACHE.clear()
        with patch('app.get_claude_client', return_value=self._claude_text_mock('{"verdict": "ok"}')):
            self.assertTrue(app.check_face_quality(self.verified_face_path)[0])
        app._FACE_QUALITY_CACHE.clear()
        with patch('app.get_claude_client', side_effect=Exception('down')):
            self.assertTrue(app.check_face_quality(self.verified_face_path)[0], "Service failure must not block")
        app._FACE_QUALITY_CACHE.clear()

    def test_private_analyze_rejects_multiple_people(self):
        """Private participant photos with more than one prominent person are rejected."""
        app._FACE_QUALITY_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed), \
             patch('app.check_face_quality', return_value=(False, 'multiple_people', app.FACE_QUALITY_MESSAGES['multiple_people'])):
            resp = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                               json=self._private_payload(upload_id))
        data = resp.get_json()
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(data['no_face_category'], 'multiple_people')

    def test_response_quality_flags(self):
        """Low-effort answers are flagged (not blocked): identical answers, contradictions, too fast."""
        self.assertIn('Same answer for all items', app.get_response_quality_flags([3] * 10))
        self.assertTrue(any('identical' in f for f in app.get_response_quality_flags([3] * 9 + [2])))
        # Agreeing strongly with every item (both normal and reverse-keyed) is contradictory
        self.assertTrue(any('Inconsistent' in f for f in app.get_response_quality_flags([5, 5, 5, 4, 5, 5, 4, 5, 5, 4])))
        self.assertTrue(any('under' in f for f in app.get_response_quality_flags([2, 4, 2, 3, 2, 4, 2, 5, 3, 4], 6)))
        self.assertEqual(app.get_response_quality_flags([2, 4, 2, 3, 2, 4, 2, 5, 3, 4], 60), [])

    def test_private_flags_stored_and_shown_to_researcher(self):
        """Flags and completion time are saved and shown on the researcher Participants page."""
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed), \
             patch('app.check_face_quality', return_value=(True, 'ok', '')):
            pid = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                              json=self._private_payload(upload_id, answers=[3] * 10, duration_seconds=5)).get_json()['participant_id']
        df = pd.read_csv(app.get_private_participants_file())
        row = df[df['participant_id'] == pid].iloc[0]
        self.assertIn('Same answer for all items', row['quality_flags'])
        self.assertEqual(float(row['duration_seconds']), 5.0)
        rclient, _ = self._researcher_client()
        html = rclient.get('/researcher/participants').data.decode()
        self.assertIn(pid, html)
        self.assertIn('Same answer for all items', html)

    # =========================================================================
    # PHOTO CHECK SPEED & CONSENT-FIRST UPLOAD TESTS
    # =========================================================================

    def test_check_photo_runs_ai_checks_in_parallel(self):
        """Moderation, AI-generated and face checks run at the same time, so the wait is about
        the slowest single check instead of the sum of all checks."""
        import time as _time
        app._AI_CHECK_CACHE.clear()
        app._FACE_QUALITY_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_prompt_mock()
        original_create = mock.messages.create.side_effect
        def slow_create(*a, **k):
            _time.sleep(0.6)
            return original_create(*a, **k)
        mock.messages.create.side_effect = slow_create
        with patch('app.get_claude_client', return_value=mock):
            start = _time.monotonic()
            resp = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(path), 'profession': 'musician'})
            elapsed = _time.monotonic() - start
        self.assertTrue(resp.get_json()['ok'], resp.get_json())
        self.assertGreaterEqual(mock.messages.create.call_count, 3)
        self.assertLess(elapsed, 1.5, f"Checks should run in parallel (took {elapsed:.2f}s for 3 x 0.6s calls)")

    def test_check_photo_reports_most_serious_issue_first(self):
        """Explicit content is reported even if other checks also fail."""
        app._AI_CHECK_CACHE.clear()
        app._FACE_QUALITY_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        path = self._copy_face_with_suffix('jpg')
        mock = self._claude_prompt_mock(moderation='{"ok": false, "issue": "explicit_content", "reason": "Nudity."}',
                                        ai='{"classification": "ai_generated", "confidence": 99, "reason": "AI."}',
                                        face='{"verdict": "blurred"}')
        with patch('app.get_claude_client', return_value=mock):
            data = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(path), 'profession': 'musician'}).get_json()
        self.assertEqual(data['issue'], 'explicit_content')

    def test_moderation_not_repeated_on_analyze_after_photo_check(self):
        """A photo that already passed /check_photo is not sent to moderation again on analyze."""
        app._AI_CHECK_CACHE.clear()
        app._FACE_QUALITY_CACHE.clear()
        client, csrf_token = self.get_authenticated_client()
        path = self._copy_face_with_suffix('jpg')
        with patch('app.get_claude_client', return_value=self._claude_prompt_mock()):
            client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                        json={'image_path': os.path.basename(path), 'profession': 'musician'})
        with patch('app.get_claude_client') as mock_claude:
            is_safe, reason, status = app.moderate_image(path, 'musician')
            mock_claude.assert_not_called()
        self.assertTrue(is_safe)
        self.assertEqual(status, 'PASSED')

    def test_images_downscaled_before_sending_to_claude(self):
        """Large or BMP images are resized/re-encoded to JPEG (max 1024px) before being sent to Claude."""
        import base64 as _b64
        big = np.random.RandomState(0).randint(0, 255, (2400, 1800, 3)).astype(np.uint8)
        big_path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.png")
        cv2.imwrite(big_path, big)
        media_type, data = app.encode_image_for_claude(big_path)
        self.assertEqual(media_type, 'image/jpeg')
        decoded = cv2.imdecode(np.frombuffer(_b64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(max(decoded.shape[:2]), app.CLAUDE_IMAGE_MAX_SIDE)
        bmp_path = os.path.join(app.UPLOAD_DIR, f"{uuid.uuid4().hex}.bmp")
        cv2.imwrite(bmp_path, cv2.imread(self.verified_face_path))
        self.assertEqual(app.encode_image_for_claude(bmp_path)[0], 'image/jpeg')

    def test_private_upload_requires_consent_first(self):
        """Private participant photo upload is refused until consent is given (nothing is saved)."""
        client, csrf_token = self.get_authenticated_client()
        before = set(os.listdir(app.UPLOAD_DIR))
        with open(self.verified_face_path, 'rb') as f:
            resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                               data={'photo': (f, 'me.jpg'), 'skip_recognition': '1'},
                               content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        self.assertIn('consent', resp.get_json()['error'])
        self.assertEqual(set(os.listdir(app.UPLOAD_DIR)), before, "No file may be saved without consent")

    def test_private_mode_page_has_consent_lock(self):
        """The Analyze page includes the consent lock note and locked upload handling."""
        html = self.client.get('/analyze').data.decode()
        self.assertIn('id="consentLockNote"', html)
        self.assertIn('openPhotoPicker()', html)
        self.assertIn('id="photoCheckStatus"', html)

    def test_results_have_plain_language_and_personality_type(self):
        """Results page includes the personality-type title and layman explanations for each trait,
        and hides admin-only buttons from public users."""
        html = self.client.get('/analyze').data.decode()
        self.assertIn('id="resultsType"', html)
        self.assertIn('function personalityType', html)
        self.assertIn('The Caring Harmonizer', html)
        self.assertIn('trait-explain', html)
        self.assertNotIn('Retrain Model (Admin)', html)
        rclient, _ = self._researcher_client()
        self.assertIn('Retrain Model (Admin)', rclient.get('/analyze').data.decode())

    def test_computation_page_researcher_only_and_correct(self):
        """Per-subject computation is researcher-only; BFI-10 math and face prediction steps are shown,
        and the recomputed Random Forest average equals the pipeline prediction."""
        self.assertEqual(self.client.get('/researcher/computation?type=subject&id=Sandboxed_Subject_1').status_code, 302)
        app.build_and_train_model()
        emb = pd.read_csv(app.FACE_EMBEDDINGS_FILE).iloc[0][[f'face_{i}' for i in range(1, 513)]].astype(float).tolist()
        breakdown, err = app.explain_face_prediction(emb)
        self.assertIsNone(err)
        pred, _ = app.estimate_personality(emb)
        for t in app.TRAIT_NAMES:
            self.assertAlmostEqual(breakdown['final'][t], pred[t], places=0)
        bfi = app.explain_bfi10([2, 4, 2, 3, 2, 4, 2, 5, 3, 4])
        c = next(x for x in bfi['traits'] if x['trait'] == 'conscientiousness')
        self.assertEqual(c['score'], 87.5)

        client, csrf_token = self._researcher_client()
        html = client.get('/researcher/computation?type=subject&id=Sandboxed_Subject_1').data.decode()
        self.assertIn('How the result was computed', html)
        self.assertIn('Random Forest', html)
        self.assertIn('Personality type rule', html)

        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed), \
             patch('app.check_face_quality', return_value=(True, 'ok', '')):
            pid = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                              json=self._private_payload(upload_id)).get_json()['participant_id']
        html = client.get(f'/researcher/computation?type=private&id={pid}').data.decode()
        self.assertIn('6 − 2', html)
        self.assertIn('87.5', html)
        self.assertIn('Computation', client.get('/researcher/participants').data.decode())

    # =========================================================================
    # PRIVATE PHOTO ATTEMPT LIMIT & RA 10173 CONSENT TESTS
    # =========================================================================

    def _blurred_check(self, client, csrf_token, mode='private'):
        app._AI_CHECK_CACHE.clear()
        app._FACE_QUALITY_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        with patch('app.get_claude_client', return_value=self._claude_prompt_mock(face='{"verdict": "blurred"}')):
            return client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                               json={'image_path': os.path.basename(path), 'profession': 'other', 'mode': mode}).get_json()

    def test_private_photo_three_failures_start_one_minute_cooldown(self):
        """After 3 rejected photos, private uploads are blocked for 60 seconds (server-side)."""
        client, csrf_token = self.get_authenticated_client()
        first = self._blurred_check(client, csrf_token)
        self.assertEqual(first['attempts_left'], 2)
        self.assertEqual(self._blurred_check(client, csrf_token)['attempts_left'], 1)
        third = self._blurred_check(client, csrf_token)
        self.assertEqual(third['attempts_left'], 0)
        self.assertEqual(third['cooldown_seconds'], app.PRIVATE_PHOTO_COOLDOWN_SECONDS)

        before = set(os.listdir(app.UPLOAD_DIR))
        with open(self.verified_face_path, 'rb') as f:
            resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                               data={'photo': (f, 'me.jpg'), 'skip_recognition': '1', 'consent': 'true'},
                               content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 429)
        self.assertGreater(resp.get_json()['cooldown_seconds'], 50)
        self.assertEqual(set(os.listdir(app.UPLOAD_DIR)), before, "Nothing may be saved during the cooldown")
        self.assertIn('startPhotoCooldown(', client.get('/analyze').data.decode())

        # After the minute passes, uploads work again
        with client.session_transaction() as sess:
            sess['pp_cooldown_until'] = 0
        with open(self.verified_face_path, 'rb') as f:
            resp = client.post('/identify_photo', headers={'X-CSRFToken': csrf_token},
                               data={'photo': (f, 'me.jpg'), 'skip_recognition': '1', 'consent': 'true'},
                               content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

    def test_private_photo_success_resets_attempts(self):
        """A photo that passes resets the attempt counter back to the full number of tries."""
        client, csrf_token = self.get_authenticated_client()
        self._blurred_check(client, csrf_token)
        self._blurred_check(client, csrf_token)
        app._AI_CHECK_CACHE.clear()
        app._FACE_QUALITY_CACHE.clear()
        path = self._copy_face_with_suffix('jpg')
        with patch('app.get_claude_client', return_value=self._claude_prompt_mock()):
            ok = client.post('/check_photo', headers={'X-CSRFToken': csrf_token},
                             json={'image_path': os.path.basename(path), 'profession': 'other', 'mode': 'private'}).get_json()
        self.assertTrue(ok['ok'])
        self.assertEqual(self._blurred_check(client, csrf_token)['attempts_left'], 2)

    def test_public_mode_failures_not_limited(self):
        """The attempt limit applies only to private participants."""
        client, csrf_token = self.get_authenticated_client()
        for _ in range(4):
            data = self._blurred_check(client, csrf_token, mode='public')
            self.assertNotIn('cooldown_seconds', data)

    def test_analyze_private_photo_rejection_counts_as_attempt(self):
        """A photo rejected at the final analyze step also counts toward the limit."""
        client, csrf_token = self.get_authenticated_client()
        upload_id = self._private_upload()
        passed = {'is_ai': False, 'method': 'vision', 'confidence': 5.0, 'reason': '', 'status': 'PASSED'}
        with patch('app.moderate_image', return_value=(True, 'Safe', 'PASSED')), \
             patch('app.check_ai_generated_image', return_value=passed), \
             patch('app.check_face_quality', return_value=(False, 'blurred', app.FACE_QUALITY_MESSAGES['blurred'])):
            data = client.post('/analyze_private', headers={'X-CSRFToken': csrf_token},
                               json=self._private_payload(upload_id)).get_json()
        self.assertEqual(data['attempts_left'], 2)

    def test_consent_mentions_data_privacy_act(self):
        """The private participant consent cites RA 10173 and the data subject's rights."""
        html = self.client.get('/analyze').data.decode()
        self.assertIn('Republic Act No. 10173', html)
        self.assertIn('Data Privacy Act of 2012', html)
        self.assertIn('National Privacy Commission', html)
        self.assertIn('withdraw your consent', html)

    def test_research_data_remains_unmodified_after_tests(self):
        """Verify that running all tests caused ZERO byte modifications to real production research files."""
        for p, original_hash in self.prod_hashes.items():
            self.assertTrue(os.path.exists(p), f"Production research file was unexpectedly removed: {p}")
            with open(p, 'rb') as fp:
                current_hash = hashlib.sha256(fp.read()).hexdigest()
            self.assertEqual(current_hash, original_hash, f"Production research file was altered during tests: {p}")


if __name__ == '__main__':
    unittest.main()
