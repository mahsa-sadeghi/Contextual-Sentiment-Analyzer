from backend.core.config import STATIC_DIR, TEMPLATES_DIR


def test_frontend_paths_are_absolute_and_exist():
    assert STATIC_DIR.is_absolute()
    assert TEMPLATES_DIR.is_absolute()
    assert STATIC_DIR.exists()
    assert TEMPLATES_DIR.exists()
