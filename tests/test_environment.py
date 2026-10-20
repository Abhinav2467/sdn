from sdn_exam_demo.environment import check_python


def test_check_python_reports_current_runtime():
    result = check_python((3, 8))
    assert result.ok is True
    assert result.detail
