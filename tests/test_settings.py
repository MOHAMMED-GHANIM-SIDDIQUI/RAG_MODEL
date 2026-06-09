from constitution_rag.settings import DATA_DIR, PDF_PATH, get_setting, is_enabled


def test_pdf_path_lives_under_data_directory():
    assert PDF_PATH.parent == DATA_DIR
    assert PDF_PATH.name == "CONSTITUTION.pdf"


def test_get_setting_prefers_supplied_secrets():
    assert get_setting("EXAMPLE", secrets={"EXAMPLE": "from-secret"}) == "from-secret"


def test_is_enabled_accepts_truthy_values():
    assert is_enabled("FLAG", secrets={"FLAG": "true"})
    assert is_enabled("FLAG", secrets={"FLAG": "1"})
    assert not is_enabled("FLAG", secrets={"FLAG": "false"})
