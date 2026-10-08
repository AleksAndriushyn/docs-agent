from evals.exact import check, present, terms


def test_terms_extraction():
    assert terms("значение THR_FS_VALUE по умолчанию 950") == ["THR_FS_VALUE", "950"]
    assert terms("абстрактные методы Init(), Loaded() и Exit()") == ["Init", "Loaded", "Exit"]
    assert terms("сначала через MAVFTP (getParamListMavftp)") == ["MAVFTP", "getParamListMavftp"]
    assert terms("Host.FDMenuMap is the FlightData map ContextMenuStrip") == \
        ["FDMenuMap", "FlightData", "ContextMenuStrip"]
    assert terms("плагины в виде .dll") == [".dll"]
    assert terms("предпочтителен режим no pulses") == []
    assert terms("газ RC выше 90%") == ["90"]  # two-letter abbreviations are prose


def test_present_is_strict_about_numbers_and_names():
    assert present("950", "по умолчанию 950 мкс")
    assert not present("950", "по умолчанию 9500")
    assert present("0.5", "через 0,5 секунды")  # Russian decimal comma
    assert not present("0.5", "через 0.55 секунды")
    assert present("THR_FS_VALUE", "параметр `thr_fs_value`")
    assert not present("THR_FS_VALUE", "параметр FS_THR_VALUE")
    assert present("SERVOx_FUNCTION", "set SERVO5_FUNCTION to 3")
    assert not present("SERVOx_FUNCTION", "set SERVO_FUNCTION to 3")


def test_check_reports_missing_terms():
    res = check(["THR_FAILSAFE = 1 включает", "порог THR_FS_VALUE"], "Установите THR_FAILSAFE в 1.")
    assert res[0]["missing"] == []
    assert res[1]["missing"] == ["THR_FS_VALUE"]
